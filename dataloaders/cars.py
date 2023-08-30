import os
import pandas as pd

import torch
from torch.utils import data
from PIL import Image

class cars(data.Dataset):
    def __init__(self, data_dir: str, split: str, transform=None, foggy=False):
        if foggy:
            print('Cityscapes foggy 11 labels')
        else:
            print('Cityscapes 11 labels')
        self.foggy = foggy
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        csv_name = os.path.join(data_dir, 'multilabels', split+'.csv')
        df_all = pd.read_csv(os.path.join(csv_name))

        # getting samples for only 11 labels
        self.classes = ['person','car', 'fence','traffic light','truck','traffic sign','bus','motorcycle','rider','train','bicycle']
        self.colors = [(220, 20, 60), (0, 0, 142), (190, 153, 153), 
                       (250, 170, 30), (0, 0, 70), (220, 220, 0), (0, 60, 100), 
                       (0, 0, 230), (255, 0, 0), (0, 80, 100), (119, 11, 32)]
        self.TRAIN_ID_TO_COLOR = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), 
                         (0, 0, 0), (111, 74, 0), (81, 0, 81), 
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), 
                         (230, 150, 140), (70, 70, 70), (102, 102, 156), 
                         (190, 153, 153), (180, 165, 180), (150, 100, 100), 
                         (150, 120, 90), (153, 153, 153), (153, 153, 153), 
                         (250, 170, 30), (220, 220, 0), (107, 142, 35), 
                         (152, 251, 152), (70, 130, 180), (220, 20, 60), 
                         (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), 
                         (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), 
                         (119, 11, 32), (0, 0, 142)]
        cols = self.classes.copy()
        cols.insert(0, 'name')
        self.df = df_all[cols]

        if foggy:
            image_dir = os.path.join(self.data_dir, 'leftImg8bit_foggy', split)
        else:
            image_dir = os.path.join(self.data_dir, 'leftImg8bit', split)

        self.image_list = []
        for path, _, files in os.walk(image_dir):
            for name in files:
                self.image_list.append(os.path.join(path, name))

        if foggy:
            self.image_list = [x for x in self.image_list if x.endswith('.01.png')]
        print('{}: Total samples'.format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def get_label(self, image_name):
        basename = os.path.basename(image_name)
        if self.foggy:
            basename = basename.replace('leftImg8bit_foggy_beta_0.01','gtFine_labelIds')
        else:
            basename = basename.replace('leftImg8bit','gtFine_labelIds')
        subdir = os.path.basename(os.path.dirname(image_name))
        df_name = os.path.join(subdir, basename)
        data = self.df.loc[self.df.name==df_name]
        return torch.tensor(data[self.classes].values)

    def __getitem__(self, index):
        image_name = self.image_list[index] 
        label = self.get_label(image_name)        
        image = Image.open(os.path.join(image_name)).convert('RGB')
        return self.transform(image), label.squeeze()