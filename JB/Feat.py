# this feature extract python script is meant to deal with i_vector (kaldi format)
# if the feature changes, this script should not work



import sys
sys.path.append("/mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v2/kaldi_io")
import kaldi_io
import numpy as np
import argparse

#give a file path(scp), return numpy arrays
def get_train_set(feats, utt2spk):
    u2s={}
    f=open(utt2spk,'r')
    for line in f:
        s=line.split()
        if len(s)==0:
            continue
        u2s[s[0]] = s[1]
    f.close()
    
    Set = {}
    for key,mat in kaldi_io.read_vec_flt_scp(feats):
        cur_ivector = mat
        cur_speaker = u2s[key]
        if cur_speaker not in Set.keys():
            Set[cur_speaker] = [mat]
        else:
            Set[cur_speaker].append(mat)

    Training_set=[]
    for k in Set.keys():
        #cast spks with too few utts
        if len(Set[k])<3:
            continue
        Training_set.append( np.array((Set[k])) )
    return Training_set

