# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03

from .duke_CamStyle import Duke
from .market_CamStyle import Market
from .dataset_loader import ImageDataset,Dataset,DADataset
from ..collate_batch import train_collate_fn, val_collate_fn
__factory = {
    'market1501': Market,
    'dukemtmc': Duke
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
