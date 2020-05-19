# Copyright(C) 2020 Liangyong Yu
import sys
sys.path.append("/mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v2/kaldi_io")
import kaldi_io
import numpy as np
from sklearn.decomposition import PCA
def EstPca(xvectors, target_energy):
    
    # pca=PCA(n_components=2)
    # reduced_x=pca.fit_transform(xvectors)
    # print(reduced_x.shape)
    ######## calculate conv 1 ################
    num_rows, num_cols = xvectors.shape
    average = 1 / num_rows * np.sum(xvectors,axis=0)
    conv = 1 / num_rows * np.dot((xvectors - average).T,xvectors - average)

    ######## calculate conv 2 ################
    # num_rows, num_cols = xvectors.shape
    # sum_ = 1 / num_rows * np.sum(xvectors,axis=0)
    # sum_ = sum_[np.newaxis, :]
    # sumsq = 1 / num_rows * np.dot(xvectors.T,xvectors)
    # conv = sumsq - np.dot(sum_.T,sum_)
    a0,b0=np.linalg.eig(conv)
    num_ = 0
    cum_ = 0
    cumsum_ = np.sum(a0)
    while cum_ < cumsum_ * target_energy:
        cum_ += a0[num_]
        num_ += 1
    transform = b0[:,:num_+1]
    return transform


