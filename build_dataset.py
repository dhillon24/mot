import os
import pandas as pd
import numpy as np
import random
import torch
from torch.functional import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        npimg = np.asarray(img, dtype=np.uint8)
        axs[0, i].imshow(npimg)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def draw_boxes(img_size, img, label, label_norm = (1,1), discard_first=True,
               transform=None):

    if not isinstance(label, Tensor):
        label = Tensor(label)

    if not isinstance(img, Tensor):
        img = np.asarray(img, dtype=np.uint8).transpose((2,0,1))
        img = Tensor(img)

    valid_boxes = label.squeeze()
    if discard_first:
        valid_boxes = valid_boxes[:, 1:]
    non_empty_mask = valid_boxes.abs().sum(dim=1).bool()
    valid_boxes = valid_boxes[non_empty_mask, :]

    valid_boxes[:, [0, 2]] *= (img_size[1] / label_norm[1])
    valid_boxes[:, [1, 3]] *= (img_size[0] / label_norm[0])

    valid_boxes[:, :2] -= valid_boxes[:, 2:] / 2.0
    valid_boxes[:, 2:] += valid_boxes[:, :2]

    if transform:
        img = transform(img)
    
    if img.dtype == torch.float32:
        img = img.to(torch.uint8)

    result = draw_bounding_boxes(img, valid_boxes, width=1)

    show(result)

class Denormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], factor=1.0):
        self.mean = mean
        self.std = std
        self.factor = factor

    def __call__(self, tensor):
        out = tensor.detach().clone()        ## make a copy
        for t, m, s in zip(out, self.mean, self.std):
            t.mul_(s).add_(m)   
        out.mul_(self.factor)
        return out
        
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, labels = None):
        
        if torch.rand(1) < self.prob:
            image = transforms.functional.hflip(image)
            if labels is not None:
                labels[..., 1] = 1.0 - labels[...,1]

        return image, labels

class RandomCrop(object):
    def __init__(self, min_ratio=0.6, max_ratio=1.0, prob=0.5, border=0.05):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prob = prob
        self.border = border

    def __call__(self, image, labels=None):

        if torch.rand(1) < self.prob:

            if not isinstance(image, np.ndarray):
                image = np.asarray(image)

            min_height = int(self.min_ratio*image.shape[0])
            min_width = int(self.min_ratio*image.shape[1])
            
            ratio = random.uniform(self.min_ratio, self.max_ratio)
            height = int(ratio*image.shape[0]) 
            width = int(ratio*image.shape[1])
            
            y = random.randint(0, image.shape[0]-height)
            x = random.randint(0, image.shape[1]-width)

            new_labels = labels.copy()

            new_labels[..., [1, 3]] *= image.shape[1]
            new_labels[..., [2, 4]] *= image.shape[0]

            min_border = int(self.border*min(height, width))

            x_max = x + width - min_border
            x_min = x + min_border

            y_max = y + height - min_border
            y_min = y + min_border

            row_indices = (new_labels[:, 1] > x_min) & (new_labels[:, 1] < x_max) & (
                new_labels[:, 2] > y_min) & (new_labels[:, 2] < y_max)

            if not row_indices.any():
                return transforms.functional.to_pil_image(image), labels

            labels = labels[row_indices,:]
            new_labels = new_labels[row_indices,:]

            new_labels[:, 1:3] -= new_labels[:, 3:] / 2.0
            new_labels[:, 3:] += new_labels[:, 1:3]

            new_labels[:, [1, 3]] -= x
            new_labels[:, [2, 4]] -= y

            new_labels[:, [1, 3]] = np.clip(new_labels[:, [1, 3]], 0, width-1)
            new_labels[:, [2, 4]] = np.clip(new_labels[:, [2, 4]], 0, height-1)

            new_labels[:, 1:3] += new_labels[:, 3:]
            new_labels[:, 1:3] *= 0.5

            new_labels[:, 3:] -= new_labels[:, 1:3]
            new_labels[:, 3:] *= 2.0

            new_labels[:, [1, 3]] /= width
            new_labels[:, [2, 4]] /= height

            labels = new_labels

            image = transforms.functional.to_pil_image(image[y:y+height, x:x+width])

        return image, labels


class CompositeCompose(object):

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, img, target):

        if self.transforms is None:
            return img, target

        for t in self.transforms:
            img, target = t(img,target)
        return img, target


class MOT20Dataset(Dataset):
    def __init__(self, seq_dir, labels_dir=None, input_transform=None, target_transform=None, 
                 composite_transform=None, img_size=(608, 1088), max_detections=300):
        super(MOT20Dataset, self).__init__()
        self.seq_dir = seq_dir
        self.labels_dir = labels_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.composite_transform = composite_transform
        self.img_filenames = {}
        self.label_filenames = {}
        self.img_filenames_registry = None
        self.label_filenames_registry = None
        self.img_size = img_size
        self.max_detections = max_detections
        self.max_tid = 0
        self.generate_labels(seq_dir, labels_dir)
        
    def __len__(self):
        return len(self.img_filenames_registry)

    def __getitem__(self, idx, img_filenames_registry=None, label_filenames_registry=None, input_transform=None,
                    target_transform=None, composite_transform=None):
        
        if img_filenames_registry is None:
            img_filenames_registry = self.img_filenames_registry
        
        if label_filenames_registry is None:
            label_filenames_registry = self.label_filenames_registry

        if input_transform is None:
            input_transform = self.input_transform

        if target_transform is None:
            target_transform = self.target_transform

        if composite_transform is None:
            composite_transform = self.composite_transform
        
        img_row = img_filenames_registry.iloc[idx, :]
        img_path = img_row.loc["img_filename"]
        img_fid = img_row.loc["fid"]

        label_row = label_filenames_registry.iloc[idx, :]
        label_path = label_row.loc["label_filename"]
        label_fid = label_row.loc["fid"]
        
        assert img_fid == label_fid
        
        label_rows = pd.read_csv(label_path, 
        names=["fid", "seq_name", "tid", "xc_norm", "yc_norm", "w_norm", "h_norm"])

        label = label_rows.loc[:, "tid":"h_norm"].to_numpy(dtype=np.float32)
        
        image = Image.open(img_path)

        if composite_transform:
            image, label = composite_transform(image, label)
        
        if input_transform:
            image = input_transform(image)
        
        label_padded = np.pad(
            label, ((0, self.max_detections - label.shape[0]), (0, 0)))
        
        if target_transform:
            label_padded = target_transform(label_padded)
        
        # draw_boxes((image.shape[1], image.shape[2]),
        #            image, label_padded, transform=Denormalize(factor=255.0))
        
        return image, label_padded

    def generate_labels(self, seq_dir, labels_dir):
        
        def find_value(str, txt, delimiter = "\n"):
            pos1 = txt.find(str)
            pos2 = pos1 + txt[pos1:].find(delimiter)
            return txt[pos1+len(str): pos2]
        
        sequences = [s for s in sorted(os.listdir(seq_dir))]
        seq_width = 0
        seq_height = 0
        seq_img_dir = ""
        seq_name = ""
        seq_img_type = ""

        tid_curr = 0
        tid_last = 0

        total_img_filenames_list = []
        total_label_filenames_list = []
        
        for sequence in sequences:

            labels_path = os.path.join(labels_dir, sequence)
            labels_subpath = os.path.join(labels_dir, sequence, "labels")
            
            if not os.path.exists(labels_subpath):
                os.makedirs(labels_subpath)
            
            seqinfo_path = os.path.join(seq_dir, sequence, "seqinfo.ini")        
            with open(seqinfo_path) as fp:        
                lines = fp.read()
                seq_width = int(find_value("imWidth=", lines))
                seq_height = int(find_value("imHeight=", lines))
                seq_img_dir = os.path.join(
                    seq_dir, sequence, find_value("imDir=", lines))
                seq_name = find_value("name=", lines)
                seq_img_type = find_value("imExt=", lines)

            img_filenames_list = []
            
            for img in sorted(os.listdir(seq_img_dir)):
                fid = int(Path(img).stem)
                img_filenames_list.append(
                    (fid, seq_name, os.path.join(seq_img_dir, img)))

            self.img_filenames[seq_name] = pd.DataFrame(img_filenames_list,
                                                        columns=["fid", "seq_name", "img_filename"])

            self.img_filenames[seq_name].to_csv(os.path.join(
                labels_path, "img_filenames.csv"), index=False)

            total_img_filenames_list.extend(img_filenames_list)

            gt_text = os.path.join(seq_dir, sequence, "gt", "gt.txt")
            gt_info = np.loadtxt(
                gt_text, dtype=np.float64, delimiter=',')
            gt_info = gt_info[np.argsort(gt_info[:, 0])]

            fid_last = int(gt_info.item((0, 0))) # First frame

            bb_list = []
            img_labels_list = []
            row = 0
            seq_tid_max = 0

            for fid, tid, x, y, w, h, mark, label, visibility in gt_info:    

                if mark == 0 and not label == 1:
                    continue;
        
                fid = int(fid)
                tid = int(tid)
                
                tid_curr = tid + tid_last   
                seq_tid_max = max(seq_tid_max, tid_curr)

                x_center = x + w/2
                y_center = y + h/2

                xc_norm = x_center / seq_width
                yc_norm = y_center / seq_height

                w_norm = w / seq_width
                h_norm = h / seq_height

                if not fid == fid_last:

                    self.write_label(labels_subpath, fid_last, img_labels_list)
                    
                    img_labels_list = []
                    fid_last = fid

                
                img_labels_list.append(
                        [fid, seq_name, tid_curr, xc_norm, yc_norm, w_norm, h_norm])
                
                row += 1

            fid_last = int(gt_info.item((-1, 0)))  # Last frame
            self.write_label(labels_subpath, fid_last, img_labels_list)

            label_filenames_list = []
            for label in sorted(os.listdir(labels_subpath)):
                fid = int(Path(label).stem)
                label_filenames_list.append(
                    (fid, seq_name, os.path.join(labels_subpath, label)))

            self.label_filenames[seq_name] = pd.DataFrame(label_filenames_list,
                                                        columns=["fid", "seq_name", "label_filename"])

            
            self.label_filenames[seq_name].to_csv(
                os.path.join(labels_path, "label_filenames.csv"), index=False)

            total_label_filenames_list.extend(label_filenames_list)
   
            tid_last = seq_tid_max
        
        if tid_last > self.max_tid:
            self.max_tid = tid_last
        
        self.img_filenames_registry = pd.DataFrame(total_img_filenames_list,
                                                    columns=["fid", "seq_name", "img_filename"])

        self.label_filenames_registry = pd.DataFrame(total_label_filenames_list,
                                                   columns=["fid", "seq_name", "label_filename"])

    def write_label(self, labels_subpath, fid, img_labels_list):
        
        labels_subpath_file = os.path.join(
                        labels_subpath, "{:06d}.txt".format(fid))

        if (os.path.exists(labels_subpath_file)):
            os.remove(labels_subpath_file)

        with open(labels_subpath_file, 'a') as f:
            label_str = ""
            for _fid, _seq_name, _tid_curr, _xc_norm, _yc_norm, _w_norm, _h_norm in img_labels_list:
               label_str += "{:d}, {}, {:d}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format(
                               _fid, _seq_name, _tid_curr, _xc_norm, _yc_norm, _w_norm, _h_norm)
            f.write(label_str)

class MOT20DatasetSubset(Dataset):

    def __init__(self, original, indices, input_transform=None, target_transform=None, composite_transform=None):
        assert isinstance(original, MOT20Dataset)
        super(MOT20DatasetSubset, self).__init__()
        self.original = original
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.composite_transform = composite_transform
        self.indices = indices
        self.img_filenames_registry = self.original.img_filenames_registry.iloc[indices, :].reset_index(drop=True)
        self.label_filenames_registry = self.original.label_filenames_registry.iloc[indices, :].reset_index(drop=True)
        img_filenames_temp = {key:[] for key in self.original.img_filenames.keys()}
        label_filenames_temp = {key:[] for key in self.original.label_filenames.keys()}
        for index in indices:
           img_row = self.original.img_filenames_registry.iloc[index, :]
           label_row = self.original.label_filenames_registry.iloc[index, :]
           img_seq = img_row["seq_name"]
           label_seq = label_row["seq_name"]
           assert img_seq == label_seq
           img_filenames_temp[img_seq].append(img_row)
           label_filenames_temp[label_seq].append(label_row)
        self.img_filenames = {key: pd.DataFrame(value).reset_index(drop=True) 
                                            for key, value in img_filenames_temp.items()}
        self.label_filenames = {key: pd.DataFrame(value).reset_index(drop=True) 
                                            for key, value in label_filenames_temp.items()}

    def __len__(self):
        return len(self.img_filenames_registry)

    def __getitem__(self, idx):
        return self.original.__getitem__(idx, img_filenames_registry=self.img_filenames_registry, 
        label_filenames_registry=self.label_filenames_registry, input_transform=self.input_transform, 
        target_transform=self.target_transform, composite_transform=self.composite_transform)
        

if __name__ == "__main__":

    np.set_printoptions(suppress=True,
                        formatter={'float_kind': '{:0.2f}'.format})

    seq_dir = os.path.join(os.path.curdir,"data", "dataset", "MOT20", "train")
    labels_dir = seq_dir

    img_size = (608, 1088)
    
    label_transforms = transforms.Compose([transforms.ToTensor()])
    image_transforms = transforms.Compose([transforms.Resize(img_size)])

    mot20 = MOT20Dataset(seq_dir, labels_dir,
                         image_transforms, label_transforms, img_size)

    trains_ids, val_ids = train_test_split(list(range(len(mot20))), test_size=0.1, random_state=99)

    mot20_train = MOT20DatasetSubset(mot20, trains_ids)
    mot20_val = MOT20DatasetSubset(mot20, val_ids)

    train_dataloader = DataLoader(mot20_train, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[24]
    label = train_labels[24]

    draw_boxes((img.shape[1], img.shape[2]), img, label)
