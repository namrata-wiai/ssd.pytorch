"""Baby Dataset Classes

Author: Namrata Deka
"""
# from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
import pdb

BABY_CLASSES = (  # always index 0
    'baby')


class BabyDetection(data.Dataset):
    """Baby Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to Baby dataset folder.
    """

    def __init__(self, root, dataset_name='Babies'):
        self.name = dataset_name
        self.root = root
        babies = glob(self.root + "/*/*_bbox.npy")

        self.images = []
        self.bboxes = []

        for baby in tqdm(babies):
            img_folder = baby.replace("_bbox.npy", "")
            images = np.array(sorted(glob(img_folder + "/*.jpg")))
            annotations = np.load(baby)

            invalid_indices = np.where(annotations.sum(axis=1)==0.0)[0]
            if len(invalid_indices != 0):
                images = list(set(images) - set(images[invalid_indices]))
                annotations = annotations[np.where(annotations.sum(axis=1) != 0.0)]

            self.images.extend(images)
            self.bboxes.extend(annotations)
        

    def __getitem__(self, index):
        im = cv2.cvtColor(cv2.imread(self.images[index]), cv2.COLOR_BGR2RGB)
        ht, wt, ch = im.shape
        im = cv2.resize(im, (300, 300))
        im = im.transpose(2, 0, 1) / 255.

        gt = np.zeros((1,5))
        gt[0,4] = 0
        gt[0,0] = self.bboxes[index][0]/wt
        gt[0,1] = self.bboxes[index][1]/ht
        gt[0,2] = self.bboxes[index][2]/wt
        gt[0,3] = self.bboxes[index][3]/ht

        gt[gt > 1] = 1
        gt[gt < 0] = 0

        # im, gt, h, w = self.pull_item(index)

        return torch.FloatTensor(im), torch.FloatTensor(gt)

    def __len__(self):
        return len(self.images)


if __name__=="__main__":
    dataset = BabyDetection(root='/scratch/data/readonly/real_data/unpaired/cleaned/anthrohospital')
    for i in range(dataset.__len__()):
        item = dataset[i]
        import pdb; pdb.set_trace()
