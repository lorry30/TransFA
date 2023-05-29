import scipy.io
import torch
import numpy as np
#import time
import os
def euclidean_dist(x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

def evaluate(probe_feature, probe_label, gallery_features, gallery_labels):
    # probe_feature = probe_feature.view(-1, 1)
    probe_feature = probe_feature.reshape((1, gallery_features.size()[1]))
    score = euclidean_dist(probe_feature, gallery_features)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)
    # index = index[::-1]
    # good index
    query_index = np.argwhere(gallery_labels==probe_label)
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp

def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()

    # find  good_index position in index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
    return ap, cmc

result = scipy.io.loadmat('./FR/FR_result.mat')
probe_feature = torch.FloatTensor(result['probe_f'])
probe_label = result['probe_label'][0]
gallery_features = torch.FloatTensor(result['gallery_f'])
gallery_labels = result['gallery_label'][0]

probe_feature = probe_feature.cuda()
gallery_features = gallery_features.cuda()

print(probe_feature.shape)

CMC = torch.IntTensor(len(gallery_labels)).zero_()
ap = 0.0
for i in range(len(probe_label)):
    ap_tmp, CMC_tmp = evaluate(probe_feature[i], probe_label[i], gallery_features, gallery_labels)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC/len(probe_label)
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(probe_label)))
with open("./FR_result.txt", 'a') as fp:
    fp.write('%f %f %f %f'%(CMC[0],CMC[4],CMC[9],ap/len(probe_label)) + '\n')