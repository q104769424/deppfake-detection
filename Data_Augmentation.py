# %%
import os
import pathlib
import shutil
import Augmentor
import torch
from torchvision import datasets
import glob


def balance_dataset(augment_dir):
    '''
    Create train folder from input augment folder
    '''

    root_dir = os.path.dirname(augment_dir)
    train_dir = os.path.join(root_dir, 'train')
    pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)
    augment_ds = datasets.ImageFolder(augment_dir)

    for image_folder in augment_ds.classes:
        full_image_folder = os.path.join(train_dir, image_folder)
        pathlib.Path(full_image_folder).mkdir(parents=True, exist_ok=True)

    '''
    Counting numbers of each class
    '''

    target = torch.tensor(augment_ds.targets)
    class_sample_count = torch.tensor([(target == t).sum()
                                       for t in torch.unique(target, sorted=True)])
    MAX = max(class_sample_count)
    sizes = torch.tensor([(MAX-size) for size in class_sample_count])
    print(sizes)

    '''
    Starting augmentation for each class
    '''

    for i, category in enumerate(augment_ds.classes):

        img_folder = os.path.join(augment_dir, category)
        output_folder = os.path.join(img_folder, 'output')
        # augment_images_folder = os.path.join(augment_dir, category)

        shutil.rmtree(output_folder, ignore_errors=True)

        p = Augmentor.Pipeline(img_folder)
        p.shear(probability=0.8, max_shear_left=3, max_shear_right=3)
        p.rotate(probability=1, max_left_rotation=3, max_right_rotation=3)
        p.flip_top_bottom(probability=0.7)

        if sizes[i] != 0:
            p.sample(sizes[i], multi_threaded=False)
            #p.sample(100, multi_threaded=False)

        '''
        Copy original and augmented files to train folders        
        '''
        dest = os.path.join(train_dir, category)

        for f in glob.glob(img_folder+"/*.jpg"):
            shutil.copy2(f, dest)

        for f in glob.glob(output_folder+"/*.jpg"):
            shutil.copy2(f, dest)


# %%
destdir = "./dstest/"
balance_dataset(os.path.join(destdir, 'augment'))
