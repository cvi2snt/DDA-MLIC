import os
import xml.etree.ElementTree as ET

from PIL import Image

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root, subset, use_difficult=False, return_difficult=False, transform=None):
        self.root = root
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.imgset_dir = os.path.join(root, 'ImageSets/Main')
        self.ann_dir = os.path.join(root, 'Annotations')
        id_list_file = os.path.join(
            self.imgset_dir, '{:s}.txt'.format(subset))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.subset = subset
        self.labels = None  # for network
        self.actual_labels = None  # for visualization
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
        self.transform = transform
        print('Loaded {} samples for {} set'.format(len(self.ids), subset))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """Returns the i-th example.

        Returns a color image and multi-label tensor. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and multi-hot encoded label tensor

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.ann_dir, id_ + '.xml'))
        # bbox = []
        label_list = []
        difficult = []
        objs = anno.findall('object')

        for obj in objs:
            # If not using difficult split, and the object is
            # difficult, skip it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            # bndbox_anno = obj.find('bndbox')
            label_list.append(obj.find('name').text.lower().strip())
        label_list = [*set(label_list)] # avoiding duplicate labels
        label_idx = [self.classes.index(x) for x in label_list]
        label_idx = torch.Tensor(label_idx).to(torch.int64)
        label = F.one_hot(label_idx, num_classes=len(self.classes)).sum(axis=0)            
        
        # Load an image
        img_file = os.path.join(self.img_dir, id_ + '.jpg')
        img = Image.open(img_file).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label