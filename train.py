import os
import argparse
import torch
import torchvision
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from build_dataset import CompositeCompose, MOT20Dataset, MOT20DatasetSubset, RandomCrop
from build_dataset import Denormalize, RandomHorizontalFlip
from detect import calculate_mAP, calculate_mAP_vectorized
from models.fairmot import FairMOT, FairMOTLoss
from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def matplotlib_imshow(img, one_channel=False, transform=None):
    if one_channel:
        img = img.mean(dim=0)
    img = img.detach()
    if transform is not None:
      img = transform(img)
    img = img.to(torch.uint8)
    img = transforms.functional.to_pil_image(img)
    npimg = np.asarray(img, dtype=np.uint8)
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(npimg)

def train_loop(dataloader, model, loss_fn, optimizer, device, writer, epoch):

  size = len(dataloader.dataset)

  for batch, (X,y) in enumerate(dataloader):
    optimizer.zero_grad()
    X = X.to(device=device)
    pred = model(X)
    y = y.to(device=device)
    losses = loss_fn(pred, y)
    mAP = calculate_mAP_vectorized(pred, y, (X.shape[2], X.shape[3]), device, model.max_detections)
    step = epoch*len(dataloader.dataset)//dataloader.batch_size + batch + 1
    for key, value in losses._asdict().items():
      writer.add_scalar("Losses/train/"+key, value, step)
    writer.add_scalar("Metrics/train/mAP", mAP, step)
    losses.total_loss.backward()
    optimizer.step()
    loss, current, precision = losses.total_loss.item(), batch * len(X), mAP.item()
    print(f"epoch: {epoch:>5d} loss: {loss:>7f} precision: {precision:>5f} [{current:>5d}/{size:>5d}]") 

def test_loop(dataloader, model, loss_fn, device, writer, epoch):

  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss = 0
  test_precision = 0

  with torch.no_grad():
    for batch, (X, y) in enumerate(dataloader):
      X = X.to(device=device)
      y = y.to(device=device)
      pred = model(X)
      losses = loss_fn(pred, y)
      mAP = calculate_mAP_vectorized(
          pred, y, (X.shape[2], X.shape[3]), device, model.max_detections)
      step = epoch*len(dataloader.dataset)//dataloader.batch_size + batch + 1
      for key, value in losses._asdict().items():
        writer.add_scalar("Losses/valid/"+key, value,step)
      writer.add_scalar("Metrics/valid/mAP", mAP, step)
      test_loss += losses.total_loss.item()
      test_precision += mAP.item()

  test_loss /= num_batches
  test_precision /= num_batches

  print(f"Test Stats: \n Epoch: {epoch:>5d} Avg Loss: {test_loss:>8f} mAP: {test_precision:>5f}\n")
  
def main(args):
  torch.manual_seed(args.manual_seed)
  torch.autograd.set_detect_anomaly(True)
  
  device = "cuda" if torch.cuda.is_available() else "cpu"

  dir_path = os.path.dirname(os.path.realpath(__file__))

  seq_dir = os.path.join(dir_path, "data", "dataset", args.dataset, "train")
  labels_dir = seq_dir

  img_size = (args.image_height, args.image_width)
  label_transforms = transforms.Compose([transforms.ToTensor()])
  image_transforms = transforms.Compose([transforms.Resize(img_size), 
                                         transforms.ColorJitter(brightness=args.brightness_jitter, hue=args.hue_jitter), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                                         ])

  composite_transforms = CompositeCompose(
      [RandomHorizontalFlip(), RandomCrop()])         # prob = 0.5, prob = 0.5

  mot20 = MOT20Dataset(seq_dir, labels_dir, image_transforms, label_transforms, composite_transforms, img_size)

  trains_ids, val_ids = train_test_split(
      list(range(len(mot20))), test_size=args.test_size, random_state=args.manual_seed)

  image_transforms_val = transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                                            ])

  mot20_train = MOT20DatasetSubset(mot20, trains_ids)    
  mot20_val = MOT20DatasetSubset(mot20, val_ids, input_transform=image_transforms_val, target_transform=label_transforms,
                                 composite_transform=CompositeCompose())

  train_dataloader = DataLoader(mot20_train, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
  
  test_dataloader = DataLoader(mot20_val, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)

  model = FairMOT((3, img_size[0], img_size[1]), device)

  summary(model, (args.batch_size, 3, img_size[0], img_size[1]))

  num_classes = mot20.max_tid

  loss_fn = FairMOTLoss(args.emb_dim, num_classes, (3, img_size[0], img_size[1]), device)

  if args.pretrained_weights is not None:
    all_states = torch.load(args.pretrained_weights)
    model.load_state_dict(all_states["state_dict"])
    if "loss_state_dict" in all_states:
      loss_fn.load_state_dict(all_states["loss_state_dict"])
    # if "optimizer" in all_states:
    #   loss_fn.load_state_dict(all_states["optimizer"])
    # if "scheduler" in all_states:
    #   loss_fn.load_state_dict(all_states["scheduler"])


  params = list(model.parameters())+list(loss_fn.parameters())
  
  optimizer = torch.optim.Adam(params,
                               lr=args.learning_rate, weight_decay=args.weight_decay)

  optimizer.zero_grad()

  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=args.step_epochs)

  save_dir = os.path.join(dir_path, "models")
  model_name = "fairmot"

  if not os.path.exists(save_dir):
      os.makedirs(save_dir)

  writer = SummaryWriter()

  it = iter(train_dataloader)
  sample_images, _ = it.next()
  sample_images = sample_images.to(device)
  writer.add_graph(model, sample_images)

  # denormalize = Denormalize(factor=255.0)
  # sample_images = denormalize(sample_images)

  # img_grid = torchvision.utils.make_grid(sample_images)
  # matplotlib_imshow(img_grid)
  # writer.add_image('sample_training_images', img_grid)

  for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}\n------------------------------")
    train_loop(train_dataloader, model, loss_fn,
               optimizer, device, writer, epoch)
    test_loop(test_dataloader, model, loss_fn, device, writer, epoch)
    state_dict = model.state_dict()
    loss_state_dict = loss_fn.state_dict()
    data = {"epoch": epoch, "state_dict": state_dict,
            "loss_state_dict": loss_state_dict}
    data["optimizer"] = optimizer.state_dict()
    data["scheduler"] = lr_scheduler.state_dict()
    torch.save(data, os.path.join(save_dir, "fairmot_{}_{}.pth".format(
        time.strftime("%Y%m%d_%H%M%S"), epoch)))
    lr_scheduler.step()

  writer.flush()
  print("Done!")
  writer.close()

if __name__ == "__main__":
  
  dir_path = os.path.dirname(os.path.realpath(__file__))
  dataset_dir = os.path.join(dir_path, "data", "dataset")
  
  if not os.path.exists(dataset_dir) or os.listdir(dataset_dir) is None:
    raise RuntimeError("Dataset directory not found or empty")
  
  dataset_list = os.listdir(dataset_dir)

  defaults = {  "dataset": "MOT20",
                "image_height": 512,
                "image_width": 512,
                "emb_dim": 128,
                "manual_seed": 24,     
                "test_size": 0.1,
                "num_workers": 4,
                "learning_rate": 1e-3,   
                "weight_decay": 1e-6,    
                "batch_size": 8,
                "epochs": 30,           
                "step": 10,           
                "brightness": 0.2,
                "hue": 0.3,
                "pretrained_weights": os.path.join(dir_path, 
                "models","fairmot_latest.pth")
            }

  parser = argparse.ArgumentParser(description='Run training for FairMOT')

  parser.add_argument("-data", "--dataset", default=defaults["dataset"],
                      help="Specify dataset from this list: {}".format(dataset_list))
  parser.add_argument("-height", "--image_height", type=int, default=defaults["image_height"],
                      help="Specify height of input images")
  parser.add_argument("-width", "--image_width", type=int, default=defaults["image_width"],
                      help="Specify width of input images")
  parser.add_argument("-embedding", "--emb_dim", type=int, default=defaults["emb_dim"],
                      help="Specify the dimension of the output ReID embedding")
  parser.add_argument("-seed", "--manual_seed", type=int, default=defaults["manual_seed"],
                      help="Specify manual random seed")
  parser.add_argument("-split", "--test_size", type=float, default=defaults["test_size"],
                      help="Specify test split ratio for training")
  parser.add_argument("-workers", "--num_workers", type=int, default=defaults["num_workers"],
                      help="Specify num of CPU workers for training")
  parser.add_argument("-rate", "--learning_rate", type=float, default=defaults["learning_rate"],
                      help="Specify the learning rate for training")
  parser.add_argument("-decay", "--weight_decay", type=float, default=defaults["weight_decay"],
                      help="Specify the weight decay factor for training")
  parser.add_argument("-batch", "--batch_size", type=int, default=defaults["batch_size"],
                      help="Specify the batch size for training")
  parser.add_argument("-epochs", "--epochs", type=int, default=defaults["epochs"],
                      help="Specify the max number of epochs for training")
  parser.add_argument("-step", "--step_epochs", type=int, default=defaults["step"],
                      help="Specify the number of epochs after which learning rate decreases by a step")
  parser.add_argument("-brightness", "--brightness_jitter", type=int, default=defaults["brightness"],
                      help="Specify the brightness jitter for color jitter transformation")
  parser.add_argument("-hue", "--hue_jitter", type=int, default=defaults["hue"],
                      help="Specify the hue jitter for color jitter transformation")
  parser.add_argument("-weights", "--pretrained_weights",
                      type=str, default=defaults["pretrained_weights"])

  args = parser.parse_args()
  
  main(args)
