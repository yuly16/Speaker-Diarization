# Copyright(C) 2020 Liangyong Yu
# This script calculates JB scores of a series of xvector. The likelihood ratio of two x-vectors is: p(x1,x2)-p(x1)-p(x2)
from JB_test import *
import sys
import multiprocessing
import os
def make_trial(tmp_file,sub_file,trial_name):
    scp =os.path.join(tmp_file,sub_file)
    trial_file = os.path.join(tmp_file,trial_name)
    f = open(scp,'r')
    utt_list = []
    for item in f.readlines():
        item = item.strip()
        utt = item.split(' ')[0]
        if not utt in utt_list:
            utt_list.append(utt)
    trial_list = []
    for i in utt_list:
        for j in utt_list:
            trial_list.append('%s %s\n'%(i,j))
    f = open(trial_file,'w')
    f.writelines(trial_list)
    f.close()
def npy2ark(npy_dir,output_dir):
    npy_list = os.listdir(npy_dir)
    npy_list.sort()
    ark_scp_output='ark:| ../../../src/featbin/copy-feats --compress=false ark:- ark,scp:%s/scores.ark,%s/scores.scp'%(output_dir,output_dir)
    with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
        for npy_file in npy_list:
            key = npy_file.split('.')[0]
            mat = np.load(os.path.join(npy_dir,npy_file))
            kaldi_io.write_mat(f, mat, key=key)
def get_utt(utt2spk):
    f = open(utt2spk,'r')
    recording_list = []
    for item in f.readlines():
        item = item.strip()
        recording = item.split(' ')[1]
        if not recording in recording_list:
            recording_list.append(recording)
    f.close()
    return recording_list

def process(data_dir,recording,tmp_file,mu_path,ep_path,transform_dir,score_file):
    print(recording)
    sub_file = 'xvector_%s_sub.scp'%(recording)
    scp =os.path.join(tmp_file,sub_file)
    trial = os.path.join(tmp_file,'trial_%s'%(recording))
    os.system("perl JB/sub_file.pl %s %s %s %s"%(recording,data_dir,tmp_file,sub_file))
    make_trial(tmp_file,'xvector_%s_sub.scp'%(recording),'trial_%s'%(recording))
    test = JB_test(mu_path, ep_path, transform_dir, scp, scp, trial)
    test.calculate_score()
    score = os.path.join(score_file,recording + '.npy')
    test.output(score)
if __name__=='__main__':
    data_dir = sys.argv[1]
    cov_dir = sys.argv[2]
    transform_dir = sys.argv[3]
    output_dir = sys.argv[4]
    mu_path = os.path.join(cov_dir,'S_mu.npy')
    ep_path = os.path.join(cov_dir,'S_ep.npy')
    tmp_file = 'exp/JB_tmp'
    score_file = os.path.join(tmp_file,'scores')
    if os.path.exists(tmp_file):
        os.system('rm -r %s'%(tmp_file))
    if not os.path.exists(output_dir):
        os.system('mkdir %s'%(output_dir))
    os.system('cp %s/spk2utt %s/'%(data_dir,output_dir))
    os.system('cp %s/utt2spk %s/'%(data_dir,output_dir))
    os.system('cp %s/segments %s/'%(data_dir,output_dir))
    os.mkdir(tmp_file)
    os.mkdir(score_file)
    recording_list = get_utt(os.path.join(data_dir,'utt2spk'))

    pool = multiprocessing.Pool(processes = 16)
    for recording in recording_list:
        #process(data_dir,recording,tmp_file,mu_path,ep_path,transform_dir,score_file)
        pool.apply_async(process, (data_dir,recording,tmp_file,mu_path,ep_path,transform_dir,score_file))
    #将npy转化为ark
    pool.close()
    pool.join()
    npy2ark(score_file,output_dir)
    



    # os.system('rm -r %s'%(tmp_file))
    # test = JB_test(sys.argv[1]+'/A.npy',sys.argv[1]+'/G.npy',sys.argv[2],sys.argv[3],sys.argv[4])
    # test.calculate_score()
    # test.output(sys.argv[5])




