from PCA import EstPca
import numpy as np
dataset='/mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v2/exp/jb/xvectors_callhome1'
# PCA(dataset + '/xvector.scp',dataset + '/JB')
transform = EstPca(dataset + '/xvector.scp',target_energy = 0.1)
