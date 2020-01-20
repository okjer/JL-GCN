# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..collate_batch import train_collate_fn


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, root ,transform=None):
        self.dataset = dataset
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fileName, pid, camid = self.dataset[index]
        img_path = osp.join(self.root,fileName)
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, fileName

class DADataset(object):
    def __init__(self,DA,root,size,transform=None):
        self.dataset = DA
        self.transform = transform
        self.root = root
        self.map = {}
        self.size = size
        for da in DA:
            fileName = da[0][0:16] + ".jpg"
            if fileName not in self.map.keys():
                self.map[fileName] = []
            self.map[fileName].append(da)
    def getBatch(self,fileName):
        data = self.map[fileName]
        dataset = ImageDataset(data,self.root,self.transform)
        loader = DataLoader(
            dataset, batch_size=self.size, shuffle=False, num_workers=0,
            collate_fn=train_collate_fn
        )
        batch = next(iter(loader))
        return batch
        
    
            