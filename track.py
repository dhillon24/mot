import os
import torch
import argparse
import lap
from torch._C import dtype
from torchvision import transforms
from torchinfo import summary
from models.fairmot import FairMOT
from detect import generate_detections_batch
from detect import intersection_over_union_vectorized
from detect import embedding_similarity_vectorized
from build_dataset import show
from PIL import Image
import cv2
import numpy as np

class MultiKalmanFilter(object):

  def __init__(self, detections, embeddings, device, dt=1/30, max_occlusion=5, std_acc=1, std_meas=(10,10), u=(1,1)):
    self.dt = dt
    self.u = torch.Tensor([[u[0], u[1]]]).to(torch.float32).to(device).view(1,2,1)
    self.std_acc = std_acc
    self.A = torch.Tensor([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]).view(1,4,4).to(device)
    self.B = torch.Tensor([[(dt**2)/2, 0], 
                          [0, (dt**2)/2],
                          [dt, 0],
                          [0, dt]]).view(1, 4, 2).to(device)
    self.H = torch.Tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0]]).view(1, 2, 4).to(device)
    self.Q = (torch.Tensor([[(dt**4)/4, 0, (dt**3)/2, 0],
                            [0, (dt**4)/4, 0, (dt**3)/2],
                            [(dt**3)/2, 0, dt**2, 0],
                            [0, (dt**3)/2, 0, dt**2]]) * std_acc**2).view(1, 4, 4).to(device)
    self.I = torch.eye(4).view(1, 4, 4).to(device)
    self.R = torch.Tensor([[std_meas[0]**2, 0],
                          [0, std_meas[1]**2]]).view(1, 2, 2).to(device)
    self.x = torch.cat((detections[:, 1:3], torch.zeros_like(detections[:, 1:3]).to(device)),dim=1).view(-1, 4, 1)
    self.P = torch.eye(4).unsqueeze(0).expand(self.x.shape[0], 4, 4).to(device)
    self.ids = detections[:, 0].to(device)
    self.embeddings = embeddings
    self.wh = detections[:, 3:].to(device)
    self.misses = torch.zeros_like(self.ids).to(device)
    self.max_occlusion = max_occlusion
    self.device = device
    

  def predict(self):
    self.x = torch.matmul(self.A, self.x) + torch.matmul(self.B, self.u)
    self.P = torch.matmul(torch.matmul(self.A, self.P), self.A.transpose(1,2)) + self.Q  

  def get_ids(self):
    return self.ids

  def delete_tracks(self, active_ids):

    active_ids = set(active_ids.cpu().numpy().flat)
    alive_mask = []
    update_mask = []
    for idx, id in enumerate(self.ids):
      if id.item() not in active_ids:
        self.misses[idx] += 1
        if self.misses[idx] > self.max_occlusion:
          alive_mask.append(0)
        else:
          alive_mask.append(1)
          update_mask.append(0)
      else:
        self.misses[idx] = 0
        alive_mask.append(1)
        update_mask.append(1)
    
    alive_mask = torch.as_tensor(alive_mask, dtype=torch.bool).to(self.device)
    self.x = self.x[alive_mask]
    self.P = self.P[alive_mask]
    self.ids = self.ids[alive_mask]
    self.embeddings = self.embeddings[alive_mask]
    self.wh = self.wh[alive_mask]
    self.misses = self.misses[alive_mask]
  
    return torch.as_tensor(update_mask, dtype=torch.bool).to(self.device)

  def update(self, detections, embeddings):
    
    active_ids = detections[:, 0]
    update_mask = self.delete_tracks(active_ids)
    
    new_mask = []
    existing_ids = set(self.ids.cpu().numpy().flat) 
    for idx, id in enumerate(active_ids):
      if id.item() not in existing_ids:
        new_mask.append(1)
      else:
        new_mask.append(0)  
    
    new_mask = torch.as_tensor(new_mask, dtype=torch.bool).to(self.device)
    existing_mask = ~new_mask

    num_existing_detections = int(existing_mask.float().sum().item())
    if num_existing_detections > 0:
      z = detections[existing_mask, 1:3].view(-1, 2, 1)
      S = torch.matmul(torch.matmul(self.H, self.P[update_mask,...]), self.H.transpose(1,2)) + self.R  
      K = torch.matmul(torch.matmul(self.P[update_mask,...], self.H.transpose(1,2)), torch.linalg.inv(S))
      self.x[update_mask,...] = self.x[update_mask,...] + torch.matmul(K, (z - torch.matmul(self.H, self.x[update_mask,...])))
      self.P[update_mask,...] = torch.matmul(self.I - torch.matmul(K, self.H), self.P[update_mask,...])
      self.embeddings[update_mask, ...] = embeddings[existing_mask, ...]
      self.wh[update_mask,...] = detections[existing_mask, 3:]
      

    num_new_detections = int(new_mask.float().sum().item())
    if num_new_detections > 0:
      new_x = torch.cat((detections[new_mask, 1:3], 
                        torch.zeros_like(detections[new_mask, 1:3]).to(self.device)), dim=1).view(-1, 4, 1)
      self.x = torch.cat((self.x, new_x))
      self.P = torch.cat((self.P,torch.eye(4).unsqueeze(0).expand(num_new_detections, 4, 4).to(self.device)))
      self.ids = torch.cat((self.ids,detections[new_mask, 0]))
      self.embeddings = torch.cat((self.embeddings, embeddings[new_mask, ...]))
      self.wh = torch.cat((self.wh,detections[new_mask, 3:]))
      self.misses = torch.cat((self.misses, torch.zeros(num_new_detections).to(self.device)))

  def get_tracks(self):
    return torch.cat((self.ids.view(-1,1), self.x[:,:2].view(-1,2), self.wh), dim=1)

  def get_embeddings(self):
    return self.embeddings  


class Tracker(object):

  def __init__(self, device, input_size, prob_thresh=0.5, min_overlap=0.1):
    self.kalman_filter = None
    self.counter = 0
    self.device = device
    self.prob_thresh = prob_thresh
    self.input_size = input_size
    self.min_overlap = min_overlap
    self.current_detections = None
    self.frame_id = 0

  def get_tracks(self):
    if self.kalman_filter is not None:
      return self.kalman_filter.get_tracks()
    else:
      raise ValueError  

  def predict(self):
    if self.kalman_filter is not None:
      self.kalman_filter.predict()

  def update(self, predictions):

    self.kalman_filter = None        ## TODO

    self.frame_id += 1
    
    heatmaps = predictions.heatmap_head.detach().clone()
    embedding_maps = predictions.reid_head.detach().clone()
    box_sizes = predictions.box_sizes_head.detach().clone()
    center_offsets = predictions.center_offsets_head.detach().clone()
    
    raw_detections, raw_embeddings = generate_detections_batch(
      self.prob_thresh, heatmaps, box_sizes, center_offsets, self.input_size, self.device, embedding_maps,
      show_heatmaps=False, with_objectness=False, generate_embeddings=True)

    raw_detections = raw_detections.squeeze(0)
    raw_embeddings = raw_embeddings.squeeze(0)

    self.current_detections = raw_detections

    if self.kalman_filter is None:
      detections = torch.cat((torch.arange(self.counter+1, self.counter+raw_detections.shape[0]+1).view(-1, 1).to(self.device),
                                       raw_detections), dim=1)
      self.counter += raw_detections.shape[0]
      self.kalman_filter = MultiKalmanFilter(
          detections, raw_embeddings, self.device)
      return

    self.predict()
    detections = self.associate(raw_detections, raw_embeddings)
    self.kalman_filter.update(detections, raw_embeddings)
    
  def associate(self, raw_detections, raw_embeddings, epsilon=0.1):
      
      previous_detections = self.kalman_filter.get_tracks()
      previous_embeddings = self.kalman_filter.get_embeddings()

      prev_corner_detections = previous_detections[:, 1:]
      prev_corner_detections[:, :2] -= prev_corner_detections[:, 2:4] / 2.0
      prev_corner_detections[:, 2:4] += prev_corner_detections[:, :2]

      raw_corner_detections = raw_detections.detach().clone()
      raw_corner_detections[:, :2] -= raw_corner_detections[:, 2:4] / 2.0
      raw_corner_detections[:, 2:4] += raw_corner_detections[:, :2]

      iou_costs = 1.0 - intersection_over_union_vectorized(prev_corner_detections, raw_corner_detections)
      embedding_costs = 1.0 - embedding_similarity_vectorized(previous_embeddings, raw_embeddings)

      infinite_mask = iou_costs.eq(1)
      infinite_costs = torch.zeros_like(iou_costs)
      infinite_costs[infinite_mask] = float('inf')

      total_costs = (iou_costs + embedding_costs+epsilon)*((~infinite_mask).float())
      total_costs += infinite_costs
      _, _, rows_assigned = lap.lapjv(total_costs.cpu().numpy(), extend_cost=True)

      prev_tracks_ids = previous_detections[:,0]

      assigned_track_ids = []
      for col, row in enumerate(rows_assigned):
        if row == -1: #or iou_costs[row, col] > (1.0 - self.min_overlap):
          self.counter += 1
          assigned_track_ids.append(self.counter)
          continue
        assigned_track_ids.append(prev_tracks_ids[row].item())

      detections = torch.cat((torch.as_tensor(
          assigned_track_ids).view(-1, 1).to(self.device), raw_detections), dim=1)

      return detections
  
  def detect_image_sequence(self, model, img_dir, transform):

    img_files = sorted(os.listdir(img_dir))

    for img_file in img_files:

      image_path = os.path.join(img_dir, img_file)

      if not os.path.isfile(image_path):
        continue

      image = Image.open(image_path)      ## read into circular buffer in another process at desired fps
      resized_image = transform(image).to(torch.float32).to(self.device).unsqueeze(0)
      predictions = model(resized_image)
      self.update(predictions)

      tracks = self.get_tracks()
      non_empty_mask = tracks[:, 1:].abs().sum(dim=1).bool()
      tracks = tracks[non_empty_mask, :]

      tracks[:, [1, 3]] *= (image.width / self.input_size[1])
      tracks[:, [2, 4]] *= (image.height / self.input_size[0])

      tracks[:, 1:3] -= tracks[:, 3:] / 2.0
      tracks[:, 3:] += tracks[:, 1:3]

      current_detections = self.current_detections

      current_detections[:, [0, 2]] *= (image.width / self.input_size[1])
      current_detections[:, [1, 3]] *= (image.height / self.input_size[0])

      current_detections[:, 0:2] -= current_detections[:, 2:] / 2.0
      current_detections[:, 2:] += current_detections[:, 0:2]
      
      cv_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

      cv2.putText(cv_image, str(self.frame_id), (int(0.025*image.width), int(0.1*image.height)), 0, 2.5, (255,255,255))
      
      for id, x1, y1, x2, y2 in tracks.cpu().numpy():
        cv2.rectangle(cv_image, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        cv2.putText(cv_image, str(id), (int(0.5*(x1+x2)-0.01*image.width),
                    int(y2-0.01*image.height)), 0, 0.3, (255, 255, 255))

      for x1, y1, x2, y2 in current_detections.cpu().numpy():
        cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
      
      cv2.imshow("tracks", cv_image)
      if cv2.waitKey(5000) & 0xFF == ord('q'):
        break;


  def detect_video(model, device, img_dir, threshold, transform):
    pass


def main(args):

  torch.manual_seed(args.manual_seed)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  torch.cuda.empty_cache()

  img_size = (args.image_height, args.image_width)
  image_transforms = transforms.Compose([transforms.Resize(img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                                         ])

  model = FairMOT((3, img_size[0], img_size[1]), device)
  all_states = torch.load(args.weights_path)
  model.load_state_dict(all_states["state_dict"])
  model.eval()
  summary(model, (1, 3, img_size[0], img_size[1]))

  kf_tracker = Tracker(device, img_size, args.detection_threshold)
  kf_tracker.detect_image_sequence(model, args.frames_dir, image_transforms)


if __name__ == "__main__":

  dir_path = os.path.dirname(os.path.realpath(__file__))
  frames_dir = os.path.join(dir_path, "frames")
  weights_path = os.path.join(dir_path, "models", "fairmot_latest.pth")

  defaults = {"frames_dir": frames_dir,
              "weights_path": weights_path,
              "image_height": 512,
              "image_width": 512,
              "manual_seed": 24,
              "detection_threshold": 0.4,
              "batch_size": 4
              }

  parser = argparse.ArgumentParser(description='Run detection for FairMOT')
  parser.add_argument("-weights", "--weights_path", default=defaults["weights_path"],
                      help="Specify the path of weights file")
  parser.add_argument("-frames", "--frames_dir", default=defaults["frames_dir"],
                      help="Specify the video frames directory")
  parser.add_argument("-height", "--image_height", type=int, default=defaults["image_height"],
                      help="Specify height of input images")
  parser.add_argument("-width", "--image_width", type=int, default=defaults["image_width"],
                      help="Specify width of input images")
  parser.add_argument("-seed", "--manual_seed", type=int, default=defaults["manual_seed"],
                      help="Specify manual random seed")
  parser.add_argument("-threshold", "--detection_threshold", type=float, default=defaults["detection_threshold"],
                      help="Specify the detection threshold above which boxes drawn")
  parser.add_argument("-batch", "--batch_size", type=int, default=defaults["batch_size"],
                      help="Specify the batch size for training")

  args = parser.parse_args()
  main(args)

