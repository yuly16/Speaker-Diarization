
import sys
sys.path.append("/mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v3/kaldi_io")
import kaldi_io
import numpy as np
import argparse
from PCA import EstPca
#suppose enroll data and test data are stored in numpy array formation
class JB_test:
    def __init__(self, mu_path, ep_path, transform_dir, enroll_feats, test_feats, trial):
        self.S_mu = np.load(mu_path)
        self.S_ep = np.load(ep_path)
        self.enroll={}
        for key,mat in kaldi_io.read_vec_flt_scp(enroll_feats):
            self.enroll[key]=mat
        self.test={}
        for key,mat in kaldi_io.read_vec_flt_scp(test_feats):
            self.test[key]=mat
        self.len = len(self.test)
        self.scores = np.zeros((self.len,self.len))
        self.trial_path=trial

        xvectors = []
        for _,mat in kaldi_io.read_vec_flt_scp(enroll_feats):
            xvectors.append(mat)
        xvectors = np.array(xvectors)
        transform = EstPca(xvectors,target_energy = 0.1)
        adapt_transform = np.array(kaldi_io.read_mat(transform_dir))
        self.adapt_transform = transform
        self.transform = transform
        self.S_mu = np.dot(np.dot(adapt_transform,self.S_mu),adapt_transform.T)
        self.S_ep = np.dot(np.dot(adapt_transform,self.S_ep),adapt_transform.T)       
        self.S_mu = np.dot(np.dot(transform.T,self.S_mu),transform)
        self.S_ep = np.dot(np.dot(transform.T,self.S_ep),transform)
        F = np.linalg.pinv(self.S_ep)
        G = np.dot(np.dot(-np.linalg.pinv(2*self.S_mu+self.S_ep),self.S_mu),F)
        self.A = np.linalg.pinv(self.S_mu+self.S_ep) - (F + G)
        self.G = G

        return
    
    def calculate_score(self):
        f=open(self.trial_path,'r')
        # self.score=[]
        # self.key=[]
        for line in f:
            s=line.split()
            if len(s)==0:
                continue
            if s[0] not in self.enroll.keys():
                print('Missing enroll key: {}'.format(s[0]))
                continue
            if s[1] not in self.test.keys():
                print('Missing test key: {}'.format(s[1]))
                continue
            e = self.enroll[s[0]]
            t = self.test[s[1]]

            e = np.dot(e,self.transform)
            t = np.dot(t,self.transform)
            
            sc = np.dot(np.dot(e,self.A),e)+np.dot(np.dot(t,self.A),t)-2*np.dot(np.dot(e,self.G),t)

            # self.key.append(s[0]+' '+s[1])
            # self.score.append(sc)
            score_row = list(self.enroll).index(s[0])
            score_column = list(self.test).index(s[1])
            self.scores[score_row,score_column] = sc

        f.close()
        return
            
            
    def output(self,file_path):
        np.save(file_path,self.scores)

        
