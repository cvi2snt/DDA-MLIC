from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from PIL import Image

class UCM_6(Dataset):
    """
    Added support for loading multiple labels
    https://stackoverflow.com/questions/56582246/correct-data-loading-splitting-and-augmentation-in-pytorch
    Apply transformations to a Dataset
    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target
    """

    def __init__(self, data_dir, transform=None, target_transform=None, csv_file=None, split=None):
        print('\n--->Loading UCM {} dataset for only 6 labels'.format(split))
        self.data_dir = data_dir        
        self.transform = transform
        self.target_transform = target_transform

        # all samples data
        df_all = pd.read_csv(csv_file, delimiter=',')

        # only 6 label's data
        df_6_all = df_all[['IMAGE\LABEL','buildings', 'cars', 'grass', 'ship', 'trees', 'water']]
        self.classes = list(df_6_all.columns[1:])
        self.n_classes = len(self.classes)

        # removing all the samples having no labels from the chosen 6
        self.df = df_6_all.loc[~(df_6_all[self.classes]==0).all(axis=1)] 
        # self.create_adj()

        # choosing the samples having only 6 labels 
        self.select_samples() 
        # print('Total {} classes: {}'.format(len(self.classes), self.classes))
        print('--> Loaded {} samples out of {}, for {} number of labels \n'.format( len(self.images), self.all_samples, len(self.classes)))
        
        # yes, you don't need these 2 lines below :(
        if transform is None and target_transform is None:
            print("Am I a joke to you? :)")
    
    def create_adj(self):
        labels = self.df.loc[:, self.df.columns != "IMAGE\LABEL"].values
        import numpy as np
        labels_t = np.transpose(labels)
        adj = np.matmul(labels_t, labels)
        nums = labels.sum(axis=0)
        my_dict = {}
        my_dict['adj'] = adj
        my_dict['nums'] = nums
        import pickle
        with open ('UCM_adj_6.pkl', 'wb') as f:
            pickle.dump(my_dict, f)
    
    def select_samples(self):

         # reduced image names having only 6 labels
        images_df = [x for x in self.df['IMAGE\LABEL'].values]

        # getting all the image names in data_dir
        images_all = list()
        for (dirpath, dirnames, filenames) in os.walk(self.data_dir):
            images_all += [os.path.join(dirpath, file) for file in filenames]

        self.images = []
        for img in images_all:
            img_temp = os.path.splitext(img)[0]
            img_temp = os.path.basename(os.path.normpath(img_temp))
            if img_temp in images_df:
                # print('This images exists in df', img_temp)
                self.images.append(img)
        self.n_samples = len(self.images)
        self.all_samples = len(images_all)
                
    def get_label(self, idx):
        filename = os.path.splitext(self.images[idx])[0]
        sample_fname = os.path.basename(os.path.normpath(filename))
        labels = self.df.loc[self.df["IMAGE\LABEL"] == sample_fname]
        labels = labels.loc[:, labels.columns != "IMAGE\LABEL"].values
        return torch.from_numpy(labels).squeeze().float()

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # getting image data
        img_name = self.images[idx]
        img = Image.open(img_name)

        # getting label data
        label = self.get_label(idx)    
        # print( target.shape)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.n_samples
   
    
class UCM(Dataset):
    """
    Added support for loading multiple labels
    https://stackoverflow.com/questions/56582246/correct-data-loading-splitting-and-augmentation-in-pytorch
    Apply transformations to a Dataset
    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target
    """

    def __init__(self, data_dir, transform=None, target_transform=None, csv_file=None, split=None):
        print('Loading UCM {} dataset'.format(split))
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv(csv_file, delimiter=',')
        self.df.pop(self.df.columns[0])
        # self.create_adj()
        self.classes = list(self.df.columns[1:])
        self.n_classes = len(self.classes)
        self.images = list()
        # import pdb; pdb.set_trace()
        for (dirpath, dirnames, filenames) in os.walk(self.data_dir):
            self.images += [os.path.join(dirpath, file) for file in filenames]
        # print('Total {} classes: {}'.format(self.n_classes, self.classes))
        print('!Loaded {} samples for {} number of labels for UCM {} dataset \n'.format(len(self.images), self.n_classes, split))
        # print(self.classes)
        # import pdb; pdb.set_trace()
        
        # yes, you don't need these 2 lines below :(
        if transform is None and target_transform is None:
            print("Am I a joke to you? :)")

    def create_adj(self):
        labels = self.df.loc[:, self.df.columns != "IMAGE\LABEL"].values
        import numpy as np
        labels_t = np.transpose(labels)
        adj = np.matmul(labels_t, labels)
        nums = labels.sum(axis=0)
        my_dict = {}
        my_dict['adj'] = adj
        my_dict['nums'] = nums
        import pickle
        with open ('UCM_adj.pkl', 'wb') as f:
            pickle.dump(my_dict, f)

    def get_label(self, idx):
        sample_fname = os.path.splitext(self.images[idx])[0]
        sample_fname = os.path.basename(os.path.normpath(sample_fname))
        labels = self.df.loc[self.df["IMAGE\LABEL"] == sample_fname]
        labels = labels[self.classes].values
        return torch.from_numpy(labels).squeeze().float()

    def get_labels(self):
        print('Using random sampling for minimzing class imbalance problem')
        # extracting only filename from the images list
        split_names=[os.path.splitext(image_name)[0] for image_name in self.images]
        names = [os.path.basename(os.path.normpath(fname)) for fname in split_names]

        # collecting gt labels
        labels = self.df.loc[self.df["IMAGE\LABEL"].isin(names)]
        return labels[self.classes].values

    def __getitem__(self, idx):
        # getting image data
        img_name = self.images[idx]
        img = Image.open(img_name).convert('RGB')

        #getting label data
        label = self.get_label(idx)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.images)

