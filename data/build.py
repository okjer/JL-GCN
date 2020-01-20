# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader
from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset,DADataset
#from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms
import os.path as osp
import torch
from scipy.spatial.distance import cdist
import numpy as np
from gcn_clustering import Feeder,gcn
import logging


def make_data_loader(cfg):
    torch.manual_seed(0)
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset1 = init_dataset(cfg.DATASETS.NAME1, root=cfg.DATASETS.ROOT_DIR1)#duke source
    dataset2 = init_dataset(cfg.DATASETS.NAME2, root=cfg.DATASETS.ROOT_DIR2)#market target

    # 源数据集
    src_num_classes = dataset1.num_train_ids
    src_train_set = ImageDataset(dataset1.train,osp.join(dataset1.images_dir,dataset1.train_path),train_transforms)
    src_train_loader = DataLoader(
        src_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    """
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        # sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
        num_workers=num_workers, collate_fn=train_collate_fn
    )"""

    src_val_set = ImageDataset(dataset1.query + dataset1.gallery,osp.join(dataset1.images_dir,dataset1.query_path),val_transforms)
    src_val_loader = DataLoader(
        src_val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    src_num_query = len(dataset1.query)
    #目标数据集
    tar_train_set = ImageDataset(dataset2.train,osp.join(dataset2.images_dir,dataset2.train_path), train_transforms)
    tar_train_loader = DataLoader(
        tar_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )

    tar_val_set = ImageDataset(dataset2.query + dataset2.gallery,osp.join(dataset2.images_dir,dataset2.query_path),val_transforms)
    tar_val_loader = DataLoader(
        tar_val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    daDataset = DADataset(dataset2.camstyle,osp.join(dataset2.images_dir,dataset2.camstyle_path),cfg.DATASETS.CamNum2-1,train_transforms)
    #daDataset.getBatch("1756_c2_f0148737")    
    return src_train_loader, src_val_loader, src_num_query, src_num_classes,tar_train_loader,tar_val_loader,daDataset

def make_gcn_trainset(cfg,model,src_train_loader,tar_train_loader,DAdataSet):
    #准备有标签样本的Feeder
    #sadasdas
    model.eval()
    feat = []
    label = []
    print(len(src_train_loader))
    with torch.no_grad():
        for i,(imgs,pids,camids,fileNames) in enumerate(src_train_loader):
            outputs = model(imgs)
            feat.extend(outputs)
            label.extend(pids.numpy())
            if i%100 == 0:
                print(i)
    label = np.array(label)
    N = len(feat)
    D = feat[0].size()[0]
    feat = torch.cat([f.view(1,D) for f in feat],0).numpy()
    
    distmat = np.power(
        cdist(feat, feat), 2).astype(np.float16)

    #distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(N, N) + \
    #            torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(N,N).t()
    #distmat.addmm_(1, -2, feat, feat.t())

    k_at_hop = cfg.GCN.K_AT_HOP
    knn_graph = np.argsort(distmat,axis = 1)[:,:k_at_hop[0]+1]

    feat_path = osp.join(cfg.OUTPUT_DIR,'feat.npy')
    knn_graph_path = osp.join(cfg.OUTPUT_DIR,'knn_graph.npy')
    label_path = osp.join(cfg.OUTPUT_DIR,'label.npy')

    np.save(feat_path,feat)
    np.save(knn_graph_path,knn_graph)
    np.save(label_path,label)

    trainset = Feeder(args.feat_path, 
                      knn_graph_path, 
                      label_path, 
                      k_at_hop)
    
    trainloader = DataLoader(
            trainset, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=cfg.DATALOADER.NUM_WORKERS, shuffle=True, pin_memory=True) 

    net = gcn().cuda()
    opt = torch.optim.SGD(net.parameters(), 1e-2, 
                          momentum=0.9, 
                          weight_decay=1e-4) 

    criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(4):
        adjust_lr(opt, epoch)
        train(trainloader, net, criterion, opt, epoch)
    return trainset

def train(loader, net, crit, opt, epoch):
    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs  = AverageMeter()
    precisions  = AverageMeter()
    recalls  = AverageMeter()

    net.train()
    end = time.time()
    for i, ((feat, adj, cid, h1id), gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(), 
                                (feat, adj, cid, h1id, gtmat))
        pred = net(feat, adj, h1id)
        labels = make_labels(gtmat).long()
        loss = crit(pred, labels)
        p,r, acc = accuracy(pred, labels)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.update(loss.item(),feat.size(0))
        accs.update(acc.item(),feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r,feat.size(0))
    
        batch_time.update(time.time()- end)
        end = time.time()
        if i % args.print_freq == 0:
            logger.info('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        data_time=data_time, losses=losses, accs=accs, 
                        precisions=precisions, recalls=recalls))

def adjust_lr(opt, epoch):
    scale = 0.1
    print('Current lr {}'.format(args.lr))
    if epoch in [1,2,3,4]:
        args.lr *=0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale
    
def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p,r,acc 