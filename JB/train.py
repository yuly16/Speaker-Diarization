from Feat import get_train_set
from JB_train import JB
import sys

if __name__=='__main__':
    
    Training_set = get_train_set(sys.argv[1],sys.argv[2])
    JointB = JB(Training_set)
    JointB.train(int(sys.argv[4]))
    JointB.Store(sys.argv[3]+'/S_mu.npy',sys.argv[3]+'/S_ep.npy')
