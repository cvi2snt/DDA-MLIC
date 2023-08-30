import os
import sys

import dataloaders as dataloaders


def get_dataset_mlic(source, target, source_dir, target_dir, train_transform, val_transform):
    print(source, target)
    if target == "foggycityscapes" and source == "cityscapes":
        t_train_dataset = dataloaders.__dict__[source](data_dir=target_dir, 
                                                       split='train', transform=train_transform, foggy=True)
        t_val_dataset = dataloaders.__dict__[source](data_dir=target_dir, 
                                                       split='val', transform=train_transform, foggy=True)
        s_train_dataset = dataloaders.__dict__[source](data_dir=source_dir, 
                                                       split='train', transform=train_transform)
        s_val_dataset = dataloaders.__dict__[source](data_dir=source_dir, 
                                                       split='val', transform=train_transform)
        class_names = s_train_dataset.classes
        num_classes = len(class_names)

    elif source == "voc" and target == "clipart":
        t_train_dataset = dataloaders.__dict__[source](data_dir=target_dir, 
                                                        subset='train', transform=train_transform)
        t_val_dataset = dataloaders.__dict__[source](data_dir=target_dir, 
                                                        subset='test', transform=val_transform)
        s_train_dataset = dataloaders.__dict__[source](data_dir=source_dir, 
                                                        subset='trainval', transform=train_transform)
        s_val_dataset = dataloaders.__dict__[source](data_dir=source_dir, 
                                                        subset='test', transform=val_transform)
        
    elif source == "clipart" and target == "voc":
        t_train_dataset = dataloaders.__dict__[target](data_dir=target_dir, 
                                                        subset='trainval', transform=train_transform)
        t_val_dataset = dataloaders.__dict__[target](data_dir=target_dir, 
                                                        subset='test', transform=val_transform)
        s_train_dataset = dataloaders.__dict__[target](data_dir=source_dir, 
                                                        subset='train', transform=train_transform)
        s_val_dataset = dataloaders.__dict__[target](data_dir=source_dir, 
                                                        subset='test', transform=val_transform)
    elif target == "DFC":     
        s_train_dataset = dataloaders.__dict__[source+'_6'](data_dir=os.path.join(source_dir, 'images_tr'), 
                                                         transform=train_transform, 
                                                         csv_file=os.path.join(source_dir, 'multilabel.csv'), 
                                                         split='train')
        s_val_dataset = dataloaders.__dict__[source+'_6'](data_dir=os.path.join(source_dir, 'images_test'), 
                                                         transform=val_transform, 
                                                         csv_file=os.path.join(source_dir, 'multilabel.csv'), 
                                                         split='val')
        t_train_dataset = dataloaders.__dict__[target](data_dir=os.path.join(target_dir, 'images_tr'), 
                                                         transform=train_transform, 
                                                         csv_file=os.path.join(target_dir, 'multilabel.csv'), 
                                                         split='train')
        t_val_dataset = dataloaders.__dict__[target](data_dir=os.path.join(target_dir, 'images_test'), 
                                                         transform=val_transform, 
                                                         csv_file=os.path.join(target_dir, 'multilabel.csv'), 
                                                         split='val')
        
    elif target == "UCM" or target == "AID":
        s_train_dataset = dataloaders.__dict__[source](data_dir=os.path.join(source_dir, 'images_tr'), 
                                                         transform=train_transform, 
                                                         csv_file=os.path.join(source_dir, 'multilabel.csv'), 
                                                         split='train')
        s_val_dataset = dataloaders.__dict__[source](data_dir=os.path.join(source_dir, 'images_test'), 
                                                         transform=val_transform, 
                                                         csv_file=os.path.join(source_dir, 'multilabel.csv'), 
                                                         split='val')
        t_train_dataset = dataloaders.__dict__[target](data_dir=os.path.join(target_dir, 'images_tr'), 
                                                         transform=train_transform, 
                                                         csv_file=os.path.join(target_dir, 'multilabel.csv'), 
                                                         split='train')
        t_val_dataset = dataloaders.__dict__[target](data_dir=os.path.join(target_dir, 'images_test'), 
                                                         transform=val_transform, 
                                                         csv_file=os.path.join(target_dir, 'multilabel.csv'), 
                                                         split='val')
    else:
        sys.exit("Incorrect dataset combination selected")
    class_names = s_train_dataset.classes
    num_classes = len(class_names)
    return s_train_dataset, s_val_dataset, t_train_dataset, t_val_dataset, num_classes, class_names