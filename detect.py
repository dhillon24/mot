import numpy as np
from numpy import double
import torch
import argparse
import os
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from models.fairmot import FairMOT
from build_dataset import draw_boxes
from torchinfo import summary
from PIL import Image

def intersection_over_union(box1, box2, epsilon = 1e-6):

  x1, y1, x2, y2 = box1[0].item(), box1[1].item(), box1[2].item(), box1[3].item()
  x3, y3, x4, y4 = box2[0].item(), box2[1].item(), box2[2].item(), box2[3].item()

  if not (x1 <= x4 and x2 >= x3 and y1 <= y4 and y2 >= y3):
    return 0.0 
  
  union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3)

  if union < epsilon:
    return 0.0

  x5 = max(x1,x3)
  y5 = max(y1,y3)
  x6 = min(x2,x4)
  y6 = min(y2,y4)

  intersection = (x6-x5)*(y6-y5)

  return intersection / (union - intersection + epsilon)


def intersection_over_union_vectorized(boxes1, boxes2, epsilon=1e-6):

  n1 = boxes1.shape[0]
  n2 = boxes2.shape[0]

  max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                     boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))

  min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                     boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))

  intersection = torch.clamp((max_xy - min_xy), min=0.0)
  intersection = intersection[:, :, 0]*intersection[:, :, 1]

  area1 = ((boxes1[:, 2] - boxes1[:, 0])*(boxes1[:, 3] -
           boxes1[:, 1])).unsqueeze(1).expand_as(intersection)
  area2 = ((boxes2[:, 2] - boxes2[:, 0])*(boxes2[:, 3] -
           boxes2[:, 1])).unsqueeze(0).expand_as(intersection)

  union = area1 + area2 - intersection

  return intersection / (union + epsilon)

def embedding_similarity_vectorized(embeddings1, embeddings2, epsilon=1e-6):

  n1 = embeddings1.shape[0]
  n2 = embeddings2.shape[0]

  assert embeddings1.shape[1] == embeddings2.shape[1]
  emb_dim = embeddings1.shape[1]

  e1 = embeddings1.unsqueeze(1).expand(n1, n2, emb_dim)
  e2 = embeddings2.unsqueeze(0).expand(n1, n2, emb_dim)

  norms1 = torch.sqrt((embeddings1**2).sum(dim=1, keepdim=True)).unsqueeze(1).expand(n1,n2,1)
  norms2 = torch.sqrt((embeddings2**2).sum(dim=1, keepdim=True)).unsqueeze(0).expand(n1,n2,1)

  similarity = (e1*e2).sum(dim=2, keepdim=True)
  cosine_similarity = torch.divide(similarity, norms1*norms2 + epsilon).squeeze(2)

  return cosine_similarity


def average_precision(detection_boxes, ground_truth_boxes, iou_threshold=0.5, epsilon=1e-6):
  """
  Expects detection boxes Nx5 and ground truth boxes Mx4

  WARNING: Vectorize method, is blowing up

  """
  
  non_empty_mask1 = detection_boxes.abs().sum(dim=1).bool()
  valid_detections = detection_boxes[non_empty_mask1, :]

  non_empty_mask2 = ground_truth_boxes.abs().sum(dim=1).bool()
  valid_ground_truths = ground_truth_boxes[non_empty_mask2, :]

  valid_detections = valid_detections[valid_detections[:,4].sort(descending=True)[1]]

  true_positives = torch.zeros((valid_detections.shape[0]))
  false_positives = torch.zeros_like(true_positives)
  total_true_boxes = valid_ground_truths.shape[0]

  assigned_boxes = [0]*total_true_boxes
  
  for box in range(valid_detections.shape[0]):
    max_iou = 0.0
    assigned_gt = -1

    for gt in range(total_true_boxes):
      iou = intersection_over_union(
          valid_detections[box, :-1].squeeze(), valid_ground_truths[gt, :].squeeze())

      if iou > max_iou:
        max_iou = iou
        assigned_gt = gt

    if max_iou > iou_threshold:
      if assigned_boxes[assigned_gt] == 0:
        true_positives[box] = 1
        assigned_boxes[assigned_gt] = 1
      else:
        false_positives[box] = 1
    else:
      false_positives[box] = 1

  true_positives_cumsum = torch.cumsum(true_positives, dim=0)
  false_positives_cumsum = torch.cumsum(false_positives, dim=0)

  recalls = true_positives_cumsum / (total_true_boxes + epsilon)
  precisions = torch.divide(true_positives_cumsum, (true_positives_cumsum + false_positives_cumsum + epsilon))

  precisions = torch.cat((torch.tensor([1]), precisions))
  recalls = torch.cat((torch.tensor([0]), recalls))

  average_precision = torch.trapz(precisions, recalls).sum()

  return average_precision  ## TODO: Calculate over different IoU thresholds and output mAP instead of AP


def average_precision_vectorized(detection_boxes, ground_truth_boxes, iou_threshold, epsilon=1e-6):
  """
  Expects detection boxes BxNx5 and ground truth boxes BxMx4
  """

  non_empty_mask1 = detection_boxes.abs().sum(dim=1).bool()
  valid_detections = detection_boxes[non_empty_mask1, :]

  non_empty_mask2 = ground_truth_boxes.abs().sum(dim=1).bool()
  valid_ground_truths = ground_truth_boxes[non_empty_mask2, :]

  valid_detections = valid_detections[valid_detections[:, 4].sort(descending=True)[
      1]]

  true_positives = torch.zeros((valid_detections.shape[0]))
  false_positives = torch.zeros_like(true_positives)
  total_true_boxes = valid_ground_truths.shape[0]
  assigned_boxes = torch.zeros((valid_ground_truths.shape[0]))

  overlaps = intersection_over_union_vectorized(
      valid_detections[:, :4], valid_ground_truths)
  best_intersection, best_gt_match = overlaps.max(1)

  for box in range(valid_detections.shape[0]):
    max_iou = best_intersection[box]
    best_gt = best_gt_match[box]

    if max_iou > iou_threshold:
      if assigned_boxes[best_gt] == 0:
        true_positives[box] = 1
        assigned_boxes[best_gt] = 1
      else:
        false_positives[box] = 1
    else:
      false_positives[box] = 1

  true_positives_cumsum = torch.cumsum(true_positives, dim=0)
  false_positives_cumsum = torch.cumsum(false_positives, dim=0)

  recalls = true_positives_cumsum / (total_true_boxes + epsilon)
  precisions = torch.divide(
      true_positives_cumsum, (true_positives_cumsum + false_positives_cumsum + epsilon))

  precisions = torch.cat((torch.tensor([1]), precisions))
  recalls = torch.cat((torch.tensor([0]), recalls))

  average_precision = torch.trapz(precisions, recalls).sum()

  # TODO: Calculate over different IoU thresholds and output mAP instead of AP
  return average_precision


def fast_2dmaxpool_nms(heatmap, kernel_size=3, prob_threshold=0.4):
  max_values = torch.nn.functional.max_pool2d(heatmap.expand(1, 1, -1, -1),
      kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze()
  equal_map = torch.eq(max_values, heatmap)
  detection_mask = equal_map.logical_and(heatmap.gt(prob_threshold))
  return detection_mask


def fast_2dmaxpool_nms_batch(heatmaps, kernel_size=3, prob_threshold=0.4):
  max_values = torch.nn.functional.max_pool2d(heatmaps,
                                              kernel_size=kernel_size, stride=1, padding=kernel_size//2)
  equal_map = torch.eq(max_values, heatmaps)
  detection_mask = equal_map.logical_and(heatmaps.gt(prob_threshold))
  return detection_mask


def generate_detections(threshold, heatmap, box_sizes, center_offsets, input_size, embedding_map=None,
                        show_heatmap=False, with_objectness=False, generate_embeddings=False, clamp=True):
  """
  generates detections in xc, yc, width, height format in input image coordinates

  """

  heatmap = heatmap.sigmoid()
  box_sizes = box_sizes.relu()

  if show_heatmap:
    hmap = heatmap.detach().cpu().numpy()
    hmap_min = np.min(hmap)
    hmap += hmap_min
    hmap_max = np.max(hmap)
    hmap /= hmap_max
    print("(Hmin, Hmax): ({},{})".format(hmap_min, hmap_max-hmap_min))
    plt.imshow(np.array(hmap*255.0, dtype=np.uint8))
    plt.show()

  detection_mask = fast_2dmaxpool_nms(heatmap, prob_threshold=threshold)
  num_detections = detection_mask.float().sum()
  detection_indices = torch.nonzero(detection_mask)
  
  num_cols = 4
  if with_objectness:
    num_cols += 1
    
  detections = torch.zeros(
      detection_indices.shape[0], num_cols, dtype=torch.float32)

  embeddings = None
  if generate_embeddings:
    if embedding_map is None:
      raise ValueError
    embeddings = torch.zeros(detection_indices.shape[0], embedding_map.shape[0], dtype=torch.float32)

  scaling_x = float(input_size[1]) / float(detection_mask.shape[1])
  scaling_y = float(input_size[0]) / float(detection_mask.shape[0])

  for idx in range(detections.shape[0]):
    row = detection_indices[idx, 0]
    col = detection_indices[idx, 1]

    detections[idx, 0] = (col + center_offsets[1, row, col])*scaling_x
    detections[idx, 1] = (row + center_offsets[0, row, col])*scaling_y
    detections[idx, 2] = box_sizes[1, row, col]
    detections[idx, 3] = box_sizes[0, row, col]

    if with_objectness:
      detections[idx, 4] = heatmap[row, col]

    if generate_embeddings:
      embeddings[idx, :] = embedding_map[:, row, col]

  if clamp:
    detections[:, [0, 2]] = torch.clamp(
        detections[:, [0, 2]], min=0, max=float(input_size[1]-1))

    detections[:, [1, 3]] = torch.clamp(
        detections[:, [1, 3]], min=0, max=float(input_size[0]-1))

  if generate_embeddings:
    return detections, embeddings
  
  return detections


def generate_detections_batch(threshold, heatmaps, box_sizes, center_offsets, input_size, device, embedding_maps = None, 
                              show_heatmaps=False, with_objectness=False, generate_embeddings=False, max_detections=-1, clamp=True):
  """
  generates detections in xc, yc, width, height format in input image coordinates

  """

  heatmaps = heatmaps.sigmoid()
  box_sizes = box_sizes.relu()

  if show_heatmaps:
    hmaps = heatmaps.detach().cpu()
    for sample in range(heatmaps.shape[0]):
      hmap = hmaps[sample].squeeze().numpy()
      hmap_min = np.min(hmap)
      hmap += hmap_min
      hmap_max = np.max(hmap)
      hmap /= hmap_max
      print("(Hmin, Hmax): ({},{})".format(hmap_min, hmap_max-hmap_min))
      plt.imshow(np.array(hmap*255.0, dtype=np.uint8))
      plt.show()

  detection_masks = fast_2dmaxpool_nms_batch(heatmaps, prob_threshold=threshold)
  num_detections_batch = detection_masks.float().sum()

  assert num_detections_batch.item() >= 1

  num_cols = 4
  if with_objectness:
    num_cols += 1

  max_detections_observed = int(detection_masks.float().view(
      detection_masks.shape[0], -1).sum(dim=1).max(dim=0)[0].item())

  if max_detections == -1 or (max_detections != -1 and max_detections_observed < max_detections):
    max_detections = max_detections_observed

  detections = torch.zeros(heatmaps.shape[0],
                           max_detections, num_cols, dtype=torch.float32).to(device)

  embeddings = None
  if generate_embeddings:
    if embedding_maps is None:
      raise ValueError
    embeddings = torch.zeros(heatmaps.shape[0],
                             max_detections, embedding_maps.shape[1], dtype=torch.float32).to(device)

  scaling_x = float(input_size[1]) / float(heatmaps.shape[-2])
  scaling_y = float(input_size[0]) / float(heatmaps.shape[-1])

  for sample in range(heatmaps.shape[0]):

    detection_rows = None
    detection_cols = None

    if detection_masks[sample].float().sum() <= max_detections:
      detection_indices = torch.nonzero(detection_masks[sample].squeeze())
      detection_rows = detection_indices[:, 0]
      detection_cols = detection_indices[:, 1]
    else:
      flattened_heatmap = (
          heatmaps[sample]*detection_masks[sample].float()).flatten()
      _, detection_indices = torch.topk(flattened_heatmap, max_detections)
      detection_rows = torch.divide(
          detection_indices, heatmaps[sample].shape[-1]).to(torch.int64)
      detection_cols = torch.remainder(
          detection_indices, heatmaps[sample].shape[-1])

    num_detections = detection_rows.shape[0]

    detections[sample, :num_detections, 0] = (detection_cols + center_offsets[sample, 1, detection_rows, detection_cols])*scaling_x
    detections[sample, :num_detections, 1] = (detection_rows + center_offsets[sample, 0, detection_rows, detection_cols])*scaling_y
    detections[sample, :num_detections, 2] = box_sizes[sample, 1, detection_rows, detection_cols]
    detections[sample, :num_detections, 3] = box_sizes[sample,0, detection_rows, detection_cols]

    if with_objectness:
        detections[sample, :num_detections, 4] = heatmaps[sample, 0, detection_rows, detection_cols]

    if generate_embeddings:
      embeddings[sample, :num_detections, :] = embedding_maps[sample,:, detection_rows, detection_cols].transpose(0,1)
  
  if clamp:
    detections[:,:,[0, 2]] = torch.clamp(
        detections[:,:,[0, 2]], min=0, max=float(input_size[1]-1))

    detections[:,:,[1, 3]] = torch.clamp(
        detections[:,:,[1, 3]], min=0, max=float(input_size[0]-1))
  
  if generate_embeddings:
    return detections, embeddings

  return detections

def calculate_mAP(predictions, ground_truth, input_size, prob_threshold=0.4, iou_threshold=0.5):

  heatmaps = predictions.heatmap_head.detach().clone()
  box_sizes = predictions.box_sizes_head.detach().clone()
  center_offsets = predictions.center_offsets_head.detach().clone()
  ground_truth = ground_truth.detach().clone()

  mean_average_precision = 0

  for sample in range(ground_truth.shape[0]):

    gt_boxes = ground_truth[sample].squeeze()
    non_empty_mask = gt_boxes.abs().sum(dim=1).bool()
    gt_boxes = gt_boxes[non_empty_mask, 1:]

    gt_boxes[:, [0, 2]] *= input_size[1]
    gt_boxes[:, [1, 3]] *= input_size[0]

    gt_boxes[:, :2] -= gt_boxes[:, 2:] / 2.0
    gt_boxes[:, 2:] += gt_boxes[:, :2]

    detections = generate_detections(
        prob_threshold, heatmaps[sample].squeeze(), box_sizes[sample], center_offsets[sample], input_size,
      show_heatmap=False, with_objectness=True)

    detections[:, :2] -= detections[:, 2:4] / 2.0
    detections[:, 2:4] += detections[:, :2]

    mean_average_precision = mean_average_precision + \
        average_precision(detections, gt_boxes, iou_threshold=iou_threshold)

  return mean_average_precision/float(ground_truth.shape[0])


def calculate_mAP_vectorized(predictions, ground_truth, input_size, device, 
                             max_detections=-1, prob_threshold=0.4, iou_threshold=0.5):

  heatmaps = predictions.heatmap_head.detach().clone()
  box_sizes = predictions.box_sizes_head.detach().clone()
  center_offsets = predictions.center_offsets_head.detach().clone()
  ground_truth = ground_truth.detach().clone()

  assert len(ground_truth.shape) == 4

  ground_truth = ground_truth.view(
      ground_truth.shape[0], ground_truth.shape[2], ground_truth.shape[3])
  ground_truth = ground_truth[:, :, 1:]
  ground_truth[:, :, [0, 2]] *= input_size[1]
  ground_truth[:, :, [1, 3]] *= input_size[0]

  ground_truth[:, :, :2] -= ground_truth[:, :, 2:] / 2.0
  ground_truth[:, :, 2:] += ground_truth[:, :, :2]

  detections = generate_detections_batch(
      prob_threshold, heatmaps, box_sizes, center_offsets, input_size, device,
      show_heatmaps=False, with_objectness=True, max_detections=max_detections)

  detections[:, :, :2] -= detections[:, :, 2:4] / 2.0
  detections[:, :, 2:4] += detections[:, :, :2]

  mean_average_precision = 0

  for sample in range(ground_truth.shape[0]):

    mean_average_precision = mean_average_precision + \
        average_precision_vectorized(
            detections[sample, :, :], ground_truth[sample, :, :], iou_threshold=iou_threshold)

  mean_average_precision /= float(ground_truth.shape[0])

  return mean_average_precision


def detect_images(model, device, img_dir, threshold, transform):

  img_files = os.listdir(img_dir)

  for img_file in img_files:
      
    image_path = os.path.join(img_dir, img_file)
    
    if not os.path.isfile(image_path):
      continue

    image = Image.open(image_path)
    resized_image = transform(image).to(torch.float32).to(device).unsqueeze(0)
    prediction = model(resized_image)

    heatmap = prediction.heatmap_head.squeeze()
    box_sizes = prediction.box_sizes_head.squeeze()
    center_offsets = prediction.center_offsets_head.squeeze()

    detections = generate_detections(
        threshold, heatmap, box_sizes, center_offsets, (resized_image.shape[2], resized_image.shape[3]),
        show_heatmap=True)

    draw_boxes((image.height, image.width), image, detections, 
                (resized_image.shape[2], resized_image.shape[3]), False)


def detect_images_batch(model, device, img_dir, threshold, transform, batch_size):

  img_files = os.listdir(img_dir)

  images = []
  resized_images = []
  
  for img_file in img_files:

    image_path = os.path.join(img_dir, img_file)

    if not os.path.isfile(image_path):
      continue

    image = Image.open(image_path)
    resized_image = transform(image).to(torch.float32).to(device).unsqueeze(0)
    
    images.append(image)
    resized_images.append(resized_image)

    if len(images) == batch_size or len(images) == len(img_files):

      input_batch = torch.cat([img for img in resized_images], dim=0)
      prediction = model(input_batch)
      heatmap = prediction.heatmap_head
      box_sizes = prediction.box_sizes_head
      center_offsets = prediction.center_offsets_head

      detections = generate_detections_batch(
          threshold, heatmap, box_sizes, center_offsets, (
              resized_image.shape[2], resized_image.shape[3]),
          device, show_heatmaps=True)

      for idx, image in enumerate(images):
        draw_boxes((image.height, image.width), image, detections[idx].squeeze(),
                    (resized_image.shape[2], resized_image.shape[3]), False)

      del detections, prediction, box_sizes, center_offsets, heatmap, input_batch
      del images, resized_images, image, resized_image

      images = []
      resized_images = []


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

  # detect_images(model, device, args.images_dir, args.detection_threshold, image_transforms)

  detect_images_batch(model, device, args.images_dir,
                args.detection_threshold, image_transforms, args.batch_size)

if __name__ == "__main__":
  
  dir_path = os.path.dirname(os.path.realpath(__file__))
  images_dir = os.path.join(dir_path, "images")
  weights_path = os.path.join(dir_path, "models", "fairmot_latest.pth")

  defaults = {"images_dir": images_dir,
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
  parser.add_argument("-images", "--images_dir", default=defaults["images_dir"],
                      help="Specify the images directory")
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
