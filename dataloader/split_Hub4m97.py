import os
global utt_id
from scipy.io import wavfile
import pandas as pd
import sys
sys.path.append("/mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v2/kaldi_io")
import kaldi_io
import numpy as np
#from generator_util import *
def split_part_file(file_name,output_dir,location):
	f = open(os.path.join(output_dir,'spk2utt'),"r")
	spk2utt_list = f.readlines()
	f.close()
	spk_list = list(map(lambda x:x.split()[0],spk2utt_list))
	f = open(os.path.join(output_dir,file_name),"r")
	reco2num_spk_list = f.readlines()
	f.close()
	reco2num_spk_list_new = []
	for item in reco2num_spk_list:
		if(item.split()[location].split('_')[0] in spk_list):
			reco2num_spk_list_new.append(item)
	f = open(os.path.join(output_dir,file_name),"w")
	f.writelines(reco2num_spk_list_new)
	f.close()

def split_file(file_name,output_dir,location):
	f = open(os.path.join(output_dir,'spk2utt'),"r")
	spk2utt_list = f.readlines()
	f.close()
	spk_list = list(map(lambda x:x.split()[0],spk2utt_list))
	f = open(os.path.join(output_dir,file_name),"r")
	reco2num_spk_list = f.readlines()
	f.close()
	reco2num_spk_list_new = []
	for item in reco2num_spk_list:
		if(item.split()[location] in spk_list):
			reco2num_spk_list_new.append(item)
	f = open(os.path.join(output_dir,file_name),"w")
	f.writelines(reco2num_spk_list_new)
	f.close()

def main():
    scp_path=sys.argv[1]
    output_dir = scp_path+'/Hub4m97'
    output_dir1 = scp_path+'/Hub4m97_1'
    output_dir2 = scp_path+'/Hub4m97_2'
    # split Hub4m97 to 2 parts
    # part1
    f = open(output_dir+'/spk2utt','r')
    lbe_files = f.readlines()
    f.close()
    lbe_files = list(map(lambda x:x.split()[0],lbe_files))
    # print(lbe_files)
    lbe_files.sort()
    lbe_files1 = lbe_files[:int(len(lbe_files)/2)]
    lbe_files2 = lbe_files[int(len(lbe_files)/2):]
    os.system("cp -r %s/spk2utt %s/"%(output_dir,output_dir1))
    os.system("cp -r %s/utt2spk %s/"%(output_dir,output_dir1))
    os.system("cp -r %s/segments %s/"%(output_dir,output_dir1))
    os.system("cp -r %s/wav.scp %s/"%(output_dir,output_dir1))
    os.system("cp -r %s/reco2num_spk %s/"%(output_dir,output_dir1))
    os.system("cp -r %s/feats.scp %s/"%(output_dir,output_dir1))
    os.system("cp -r %s/vad.scp %s/"%(output_dir,output_dir1))
    f = open(os.path.join(output_dir1,'spk2utt'),'r')
    spk2utt = f.readlines()
    f.close()
    spk2utt_new = []
    for spk2utt_item in spk2utt:
        i = spk2utt_item.split()[0]
        if(i in lbe_files1):
            spk2utt_new.append(spk2utt_item)
    f = open(os.path.join(output_dir1,'spk2utt'),'w')
    f.writelines(spk2utt_new)
    f.close()	
    # os.system("utils/fix_data_dir.sh data/Hub4m97_1")
    split_file('wav.scp',output_dir1,location=0)
    split_file('segments',output_dir1,location=1)
    split_file('utt2spk',output_dir1,location=1)
    split_file('reco2num_spk',output_dir1,location=0)
    split_part_file('feats.scp',output_dir1,location=0)
    split_part_file('vad.scp',output_dir1,location=0)
    # part2
    os.system("cp -r %s/spk2utt %s/"%(output_dir,output_dir2))
    os.system("cp -r %s/utt2spk %s/"%(output_dir,output_dir2))
    os.system("cp -r %s/segments %s/"%(output_dir,output_dir2))
    os.system("cp -r %s/wav.scp %s/"%(output_dir,output_dir2))
    os.system("cp -r %s/reco2num_spk %s/"%(output_dir,output_dir2))
    os.system("cp -r %s/feats.scp %s/"%(output_dir,output_dir2))
    os.system("cp -r %s/vad.scp %s/"%(output_dir,output_dir2))
    f = open(os.path.join(output_dir2,'spk2utt'),'r')
    spk2utt = f.readlines()
    f.close()
    spk2utt_new = []
    for spk2utt_item in spk2utt:
        i = spk2utt_item.split()[0]
        if(i in lbe_files2):
            spk2utt_new.append(spk2utt_item)
    f = open(os.path.join(output_dir2,'spk2utt'),'w')
    f.writelines(spk2utt_new)
    f.close()	
    # os.system("utils/fix_data_dir.sh data/Hub4m97_2")
    split_file('wav.scp',output_dir2,location=0)
    split_file('segments',output_dir2,location=1)
    split_file('utt2spk',output_dir2,location=1)
    split_file('reco2num_spk',output_dir2,location=0)
    split_part_file('feats.scp',output_dir2,location=0)
    split_part_file('vad.scp',output_dir2,location=0)
if __name__ == "__main__":
	main()
