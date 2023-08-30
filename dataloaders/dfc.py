from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from PIL import Image

class DFC(Dataset):

    """DFC_15 ML dataset."""

    def __init__(self, data_dir, transform=None, target_transform=None, csv_file=None, split=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print('---->Loading DFC_15 {} dataset for only 6 labels'.format(split))
        self.root_dir = data_dir
        self.transform = transform

        # all samples data
        df_all = pd.read_csv(csv_file)

        # only 6 labels related data
        df_6_all = df_all[['image\label','building', 'car', 'vegetation', 'boat', 'tree', 'water']]
        self.classes = list(df_6_all.columns[1:])

        # removing all the samples having no labels from the chosen 6
        self.df = df_6_all.loc[~(df_6_all[self.classes]==0).all(axis=1)] 
        # import pdb; pdb.set_trace()

        # self.create_adj()

        # choosing the samples having only 6 labels 
        self.select_samples() 
        # print('Total {} classes: {}'.format(len(self.classes), self.classes))
        print('--> Loaded {} samples out of {}, for {} number of labels \n'.format( len(self.images), len(os.listdir(self.root_dir)), len(self.classes)))


    
    def create_adj(self):
        labels = self.df.loc[:, self.df.columns != "image\label"].values
        import numpy as np
        labels_t = np.transpose(labels)
        adj = np.matmul(labels_t, labels)
        nums = labels.sum(axis=0)
        my_dict = {}
        my_dict['adj'] = adj
        my_dict['nums'] = nums
        import pickle
        with open ('DFC_adj_6.pkl', 'wb') as f:
            pickle.dump(my_dict, f)

    def select_samples(self):

        # all the sample names from data dir
        images_all = os.listdir(self.root_dir)
        images_all = [os.path.splitext(x)[0] for x in images_all]  

        # reduced images names having only 6 labels
        images_df = self.df['image\label'].values
        images_df = list(images_df)
        images_df = [str(x) for x in images_df]

        # finding common samples from data_dir
        new_images = list(set(images_df).intersection(images_all))
        self.images = [x+'.png' for x in new_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # getting image data
        img_name = os.path.join(self.root_dir,self.images[idx])
        img = Image.open(img_name)

        # getting label data
        label = self.get_label(idx)        

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_label(self, idx):
        filename = int(os.path.splitext(self.images[idx])[0])
        labels = self.df.loc[self.df["image\label"] == filename]
        labels = labels.loc[:, labels.columns != "image\label"].values
        return torch.from_numpy(labels).squeeze().float()
