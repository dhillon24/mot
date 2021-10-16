import math
import numbers
from matplotlib.pyplot import xscale
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple

class Conv2dBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, momemtum=0.1):
    super(Conv2dBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels, momentum=momemtum)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):  
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    return x


class Deconv2dBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=1, momemtum=0.1):
    super(Deconv2dBlock, self).__init__()
    self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                                      dilation=dilation, output_padding=output_padding, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels, momentum=momemtum)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.deconv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    return x


class HeadBlock(nn.Module):

  def __init__(self, in_channels, intermediate_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, use_output_sigmoid=False):
    super(HeadBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=intermediate_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                           bias=True)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(in_channels=intermediate_channels,
                           out_channels=out_channels, kernel_size=1, padding=0, bias=True)
    self.sigmoid2 = nn.Sigmoid()
    self.use_output_sigmoid = use_output_sigmoid
    self.epsilon = 1e-4
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    if self.use_output_sigmoid:
      x = self.sigmoid2(x)
      x = torch.clamp(x, min=self.epsilon, max=1.0 - self.epsilon)
    return x

class FairMOT(nn.Module):

  def __init__(self, input_shape, device, max_detections=300, subsampling=4):
    super(FairMOT, self).__init__()

    def idx(row, col):
      return col+row*self.subsampling
    
    self.device = device
    self.subsampling = subsampling
    self.input_shape = input_shape
    self.max_detections = max_detections

    in_channels = input_shape[0]
    out_channels = 32
    max_channels = 256
    kernel_size = 3
    stride = 4
    padding = 1
    
    channels = {}
    setattr(self, "downsample_input_0_0", Conv2dBlock(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    channels[0] = out_channels
    in_channels = out_channels
    
    stride = 2

    for row in range(1, subsampling):                  
      col = row
      name = "downsample_{:n}_{:n}_{:n}_{:n}".format(row-1, col-1, row, col)
      setattr(self, name, Conv2dBlock(in_channels=in_channels,
                    out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
      in_channels = out_channels
      channels[idx(row, col)] = out_channels
      
    for row in range(subsampling-1):
      num_channels = channels[idx(row,row)]
      for col in range(row+1,subsampling):
        num_channels *= 2
        channels[idx(row,col)] = num_channels

    inc = 0
    col = 0   
    for level in range(subsampling-1,0,-1):          
      for row in range(level):                                
        col = row + 1 + inc             
        
        in_channels = channels[idx(row, col-1)]
        out_channels = channels[idx(row, col)]

        setattr(self, "same_{:n}_{:n}_{:n}_{:n}".format(row, col-1, row, col), Conv2dBlock(in_channels=in_channels,
                                                                                           out_channels=out_channels, kernel_size=kernel_size))
        
        in_channels = channels[idx(row+1, col)]
        
        setattr(self, "upsample_{:n}_{:n}_{:n}_{:n}".format(row+1, col, row, col), Deconv2dBlock(in_channels=in_channels,
                                                                                           out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                                                           padding=padding, output_padding=padding))
        in_channels = 2*out_channels
        
        setattr(self, "aggregrate_{:n}_{:n}".format(row, col), Conv2dBlock(in_channels=in_channels,
                                                                           out_channels=out_channels, kernel_size=kernel_size))
      inc += 1

    upsampling_kernel_size = kernel_size 
    upsampling_padding = padding 
    upsampling_stride = stride
    in_channels = channels[idx(0,subsampling-1)]
    out_channels = in_channels
    
    for level in range(subsampling-1):
      
      last_aggregrated_channels = channels[idx(level+1,subsampling-1)]
      
      setattr(self, "same_post_" + str(level), Conv2dBlock(in_channels=in_channels,
              out_channels=out_channels, kernel_size=kernel_size))
      setattr(self, "upsample_{:n}_{:n}_post_{:n}".format(level+1, col, level), Deconv2dBlock(in_channels=last_aggregrated_channels,
              out_channels=last_aggregrated_channels, kernel_size=upsampling_kernel_size, stride=upsampling_stride, padding=upsampling_padding, output_padding=padding))
      setattr(self, "aggregrate_post_" + str(level), Conv2dBlock(in_channels=out_channels+last_aggregrated_channels,
              out_channels=out_channels+last_aggregrated_channels, kernel_size=kernel_size))

      in_channels = out_channels+last_aggregrated_channels
      out_channels = in_channels
      
      upsampling_kernel_size = 2*upsampling_kernel_size + 1  
      upsampling_padding *= 2                                
      upsampling_stride *= 2

    head_channels = 256
    self.heatmap_head = HeadBlock(in_channels=in_channels, intermediate_channels=head_channels, out_channels=1, kernel_size=kernel_size,
                                  use_output_sigmoid=False)
    self.box_sizes_head = HeadBlock(in_channels=in_channels, intermediate_channels=head_channels,
                                    out_channels=2, kernel_size=kernel_size, use_output_sigmoid=False)
    self.center_offsets_head = HeadBlock(
        in_channels=out_channels, intermediate_channels=head_channels, out_channels=2, kernel_size=kernel_size, use_output_sigmoid=False)
        
    head_channels = 128
    self.reid_head = nn.Conv2d(in_channels=in_channels,
                               out_channels=head_channels, kernel_size=kernel_size, padding=padding, bias=True)

    super(FairMOT, self).to(device)

  def forward(self, x):
    
    def idx(row, col):
      return col+row*self.subsampling;
    
    partial_outputs = {}

    for row in range(self.subsampling):
      col = row
      name = "downsample_{:n}_{:n}_{:n}_{:n}".format(
          row-1, col-1, row, col) if row > 0 else "downsample_input_0_0"
      op = getattr(self, name)
      x = op(x)
      partial_outputs[row] = x

    inc = 0
    col = 0
    for level in range(self.subsampling-1, 0, -1):
      for row in range(level):
        col = row + 1 + inc

        op1 = getattr(self, "same_{:n}_{:n}_{:n}_{:n}".format(row, col-1, row, col))
        op2 = getattr(self, "upsample_{:n}_{:n}_{:n}_{:n}".format(row+1, col, row, col))
        op3 = getattr(self, "aggregrate_{:n}_{:n}".format(row, col))

        partial_outputs[row] = op3(
            torch.cat((op1(partial_outputs[row]), op2(partial_outputs[row+1])), 1))

      inc += 1

    x = partial_outputs[0]
    for level in range(self.subsampling-1):
      op1 = getattr(self, "same_post_" + str(level))
      op2 = getattr(self, "upsample_{:n}_{:n}_post_{:n}".format(level+1, col, level))
      op3 = getattr(self, "aggregrate_post_" + str(level))
      x = op3(torch.cat((op1(x), op2(partial_outputs[level+1])),1))
      partial_outputs[level+1] = 0
      torch.cuda.empty_cache()


    out_dict = {}

    out_dict["heatmap_head"] = self.heatmap_head(x)
    out_dict["box_sizes_head"] = self.box_sizes_head(x)
    out_dict["center_offsets_head"] = self.center_offsets_head(x)
    out_dict["reid_head"] = self.reid_head(x)

    out_named_tuple = namedtuple("ModelEndpoints", sorted(out_dict.keys()))
    out = out_named_tuple(**out_dict)

    return out

class FocalLoss(nn.Module):
  def __init__(self, alpha=2, beta=4):
    super(FocalLoss, self).__init__()
    self.alpha=2                            # alpha=2
    self.beta=4

  def forward(self, predicted_output, ground_truth):
    
    positive_indices = ground_truth.eq(1).float()
    negative_indices = ground_truth.lt(1).float()

    negative_weights = torch.pow(1 - ground_truth, self.beta)
    
    positive_loss = torch.pow(1-predicted_output, self.alpha)*torch.log(predicted_output)*positive_indices
    negative_loss = negative_weights*torch.pow(predicted_output, self.alpha)*torch.log(1-predicted_output)*negative_indices  

    loss = 0

    num_positives = positive_indices.sum()
    positive_loss = positive_loss.sum()
    negative_loss = negative_loss.sum()

    if num_positives >= 1.0:
      loss = loss  - (positive_loss + negative_loss) / num_positives
    else:
      loss = loss  - negative_loss

    return loss

class StaticGaussianSmoothing(nn.Module):

  def __init__(self, channels, kernel_size, sigma, dim=2):

    super(StaticGaussianSmoothing, self).__init__()

    if isinstance(kernel_size, numbers.Number):
      kernel_size = [kernel_size]*dim

    if isinstance(sigma, numbers.Number):
      sigma = [sigma]*dim

    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size])

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
      mean = (size - 1) /2 
      kernel *= torch.exp((-((mgrid - mean) / std) ** 2) / 2)

    # kernel = kernel / kernel.sum()

    kernel = kernel.view(1,1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    self.register_buffer("weight", kernel)
    self.groups = channels

    if dim == 1:
      self.conv = F.conv1d
    elif dim == 2:
      self.conv = F.conv2d
    elif dim == 3:
      self.conv = F.conv3d
    else:
      raise RuntimeError('Expected upto 3 dimensions, Invalid Dimensions Received {}.'.format(dim))

  def forward(self, x):
    return self.conv(x, weight=self.weight, groups=self.groups, padding="same")


class FairMOTLoss(nn.Module):

  def __init__(self, emb_dim, num_classes, input_shape, device, std=3):
    super(FairMOTLoss, self).__init__()

    self.input_shape = input_shape
    self.emb_dim = emb_dim
    self.num_classes = num_classes
    self.classifier = nn.Linear(emb_dim, num_classes)
    self.smoother = StaticGaussianSmoothing(channels=1, kernel_size=5, sigma=std,  dim=2)
    self.reid_loss = nn.CrossEntropyLoss()
    self.heatmap_loss = FocalLoss()
    self.box_sizes_loss = torch.nn.L1Loss(reduction="sum") # torch.nn.HuberLoss(reduction='sum')
    self.center_offsets_loss = torch.nn.L1Loss(reduction="sum") # torch.nn.HuberLoss(reduction='sum')
    self.heatmap_loss_weight = 1.0              # 1.0
    self.box_sizes_loss_weight = 0.1            # 0.1
    self.center_offsets_loss_weight = 1.0       # 1.0
    self.reid_loss_weight = 1.0                 # 1.0        
    self.detection_loss_learnable_weight = 1.0 # nn.Parameter(-1.85*torch.ones(1))
    self.reid_loss_learnable_weight = 1.0 # nn.Parameter(-1.05*torch.ones(1))
    self.device = device
    super(FairMOTLoss, self).to(device)

  def forward(self, predicted_outputs, ground_truth):

    def relu(x):
      return torch.clamp(x.relu(), min=1e-4)
    
    def sigmoid(x):
      return torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)

    def clamp(x, upper, lower=0):
      return upper - 1 if x >= upper else lower if x <= lower else x
    
    heatmap_loss_sum = 0
    box_sizes_loss_sum = 0
    center_offsets_loss_sum = 0
    reid_loss_sum = 0
    det_loss_sum = 0
    loss = 0

    predicted_heatmaps = sigmoid(predicted_outputs.heatmap_head)
    predicted_box_sizes = predicted_outputs.box_sizes_head
    predicted_center_offsets = predicted_outputs.center_offsets_head
    predicted_reid_features = predicted_outputs.reid_head

    map_size = predicted_heatmaps.shape[2:]
    image_size = self.input_shape[1:]

    detection_masks = torch.zeros_like(
        predicted_heatmaps, dtype=torch.bool).to(self.device)

    gt_box_sizes = torch.zeros_like(
        predicted_box_sizes, dtype=torch.float32).to(self.device)

    gt_center_offsets = torch.zeros_like(
        predicted_center_offsets, dtype=torch.float32).to(self.device)
    
    for sample in range(ground_truth.shape[0]):
    
      valid_boxes = ground_truth[sample].squeeze()
      non_empty_mask = valid_boxes.abs().sum(dim=1).bool()
      valid_boxes = valid_boxes[non_empty_mask, :]

      # detection_mask = torch.zeros(*map_size, dtype=torch.bool).to(self.device)
      # gt_box_sizes = torch.zeros(2, *map_size, dtype=torch.float32).to(self.device)
      # gt_center_offsets = torch.zeros(2, *map_size, dtype=torch.float32).to(self.device)

      gt_class_labels = torch.zeros(valid_boxes.shape[0], dtype=torch.int64).to(self.device)
      predicted_class_labels = torch.zeros(
          valid_boxes.shape[0], self.num_classes, dtype=torch.float32).to(self.device)

      for box in range(valid_boxes.shape[0]):

        tid = valid_boxes[box, 0]
        class_id = torch.Tensor([[self.num_classes - 1]]).to(self.device) \
                   if valid_boxes[box, 0].item() > self.num_classes \
                   else valid_boxes[box, 0] - 1 
        xc = valid_boxes[box, 1].item()
        yc = valid_boxes[box, 2].item()
        w = valid_boxes[box, 3].item()
        h = valid_boxes[box, 4].item()

        yc_output = yc*map_size[0]
        xc_output = xc*map_size[1]
        mask_row_i = int(yc*map_size[0])
        mask_col_i = int(xc*map_size[1])

        mask_row_i = clamp(mask_row_i, map_size[0])
        mask_col_i = clamp(mask_col_i, map_size[1])
        
        gt_box_sizes[sample, 0, mask_row_i, mask_col_i] = h*image_size[0]
        gt_box_sizes[sample, 1, mask_row_i, mask_col_i] = w*image_size[1]
        gt_center_offsets[sample, 0, mask_row_i, mask_col_i] = yc_output - int(yc_output)
        gt_center_offsets[sample, 1, mask_row_i, mask_col_i] = xc_output - int(xc_output)
        
        detection_masks[sample, 0, mask_row_i, mask_col_i] = True

        predicted_class_labels[box, :] = self.classifier(predicted_reid_features[sample,
                                                    :, mask_row_i, mask_col_i].squeeze()).unsqueeze(0)

        gt_class_labels[box] = class_id

      # gt_heatmap = self.smoother(detection_mask.float().view(1,1,*detection_mask.shape))

      # heatmap_loss_sum = heatmap_loss_sum + self.heatmap_loss(
          # predicted_heatmaps[sample].squeeze(), gt_heatmap.squeeze()) 
      
      # box_sizes_loss_sum = box_sizes_loss_sum + self.box_sizes_loss(
      #     predicted_box_sizes[sample, :, detection_mask].squeeze(), gt_box_sizes[:, detection_mask].squeeze())

      # center_offsets_loss_sum = center_offsets_loss_sum + self.center_offsets_loss(
          # predicted_center_offsets[sample, :, detection_mask].squeeze(), gt_center_offsets[:, detection_mask].squeeze())

      reid_loss_sum = reid_loss_sum + self.reid_loss_weight*self.reid_loss(
          predicted_class_labels, gt_class_labels)    

    gt_heatmaps = self.smoother(detection_masks.float())

    heatmap_loss_sum = heatmap_loss_sum + self.heatmap_loss(predicted_heatmaps, gt_heatmaps)
    
    detection_masks = detection_masks.expand(-1, 2, -1, -1)
    
    box_sizes_loss_sum = box_sizes_loss_sum + self.box_sizes_loss(
        predicted_box_sizes[detection_masks].contiguous(), gt_box_sizes[detection_masks].contiguous())
    
    center_offsets_loss_sum = center_offsets_loss_sum + self.center_offsets_loss(
        predicted_center_offsets[detection_masks].contiguous(), gt_center_offsets[detection_masks].contiguous())
    
    det_loss_sum = det_loss_sum + self.heatmap_loss_weight*heatmap_loss_sum + self.box_sizes_loss_weight*box_sizes_loss_sum + \
               self.center_offsets_loss_weight*center_offsets_loss_sum

    # loss = torch.exp(-self.detection_loss_learnable_weight)*det_loss_sum + \
    #     torch.exp(-self.reid_loss_learnable_weight)*reid_loss_sum + \
    #     self.detection_loss_learnable_weight + self.reid_loss_learnable_weight

    loss = loss + self.detection_loss_learnable_weight*det_loss_sum + \
        self.reid_loss_learnable_weight*reid_loss_sum 

    loss_dict = {"total_loss": loss, "heatmap_loss": heatmap_loss_sum,
                 "box_sizes_loss": box_sizes_loss_sum, "center_offsets_loss": center_offsets_loss_sum,
                 "detection_loss": det_loss_sum, "reid_loss": reid_loss_sum}

    loss_named_tuple = namedtuple("ModelLosses", sorted(loss_dict.keys()))
    losses = loss_named_tuple(**loss_dict)

    return losses             
