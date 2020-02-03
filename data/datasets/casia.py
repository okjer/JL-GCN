from __future__ import print_function, absolute_import
import os.path as osp
import os
import numpy as np
import pdb
from glob import glob
import re


class CASIA(object):

    def __init__(self, root):
        self.images_dir = root
        self.train = []
        self.num_train_ids = 0
        self.load()

    def preprocess(self, relabel=True):
        #pattern = re.compile(r'([-\d]+)_c(\d)')
        ret = []
        pids = os.listdir(self.images_dir)#
        for i,pid in enumerate(pids):
            fnames = os.listdir(osp.join(self.images_dir,pid))
            for fname in fnames:
                ret.append((osp.join(str(pid),fname),int(pid),0))
        return ret, i+1

    def load(self):
        self.train, self.num_train_ids = self.preprocess()

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))

