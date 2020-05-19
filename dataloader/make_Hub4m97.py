# Copyright(C) 2020 Liangyong Yu
# This script processes raw Hub4m97 dataset into kaldi format. 
# wav.scp, segments, utt2spk, spk2utt, ref.rttm and reco2num_spk are generated for speaker diarization.
import os
global utt_id
from scipy.io import wavfile
import pandas as pd
import sys
sys.path.append("/mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v2/kaldi_io")
import kaldi_io
import numpy as np

def fix_reco2num_spk(output_dir):
	f = open(os.path.join(output_dir,'spk2utt'),"r")
	spk2utt_list = f.readlines()
	f.close()
	spk_list = list(map(lambda x:x.split()[0],spk2utt_list))
	f = open(os.path.join(output_dir,'reco2num_spk'),"r")
	reco2num_spk_list = f.readlines()
	f.close()
	reco2num_spk_list_new = []
	for item in reco2num_spk_list:
		if(item.split()[0] in spk_list):
			reco2num_spk_list_new.append(item)
	f = open(os.path.join(output_dir,'reco2num_spk'),"w")
	f.writelines(reco2num_spk_list_new)
	f.close()

def write_reco2num_spk(lbe_dir,output_dir):
	pd_info = {'reco_id':[],'num_spk':[]}
	for lbe_file in os.listdir(lbe_dir):
		lbe_name = lbe_file.split('.')[0]
		lbe_path = os.path.join(lbe_dir,lbe_file)
		f = open(lbe_path,"rb")
		lbe_lines = f.readlines()
		f.close()
		spk_num = int(lbe_lines[0].split(b" ")[1].decode("UTF-8").split("=")[1])
		false_spk_num = int(lbe_lines[0].split(b" ").count(b"?"))
		pd_info['reco_id'].append(lbe_name)
		pd_info['num_spk'].append(spk_num-false_spk_num)

	df = pd.DataFrame(pd_info)
	df = df.sort_values('reco_id',ascending=True)
	df.to_csv(os.path.join(output_dir,'reco2num_spk'),sep=' ',index=False,header=False)
def write_rttm(lbe_dir,output_dir):
	pd_info = {'PREFIX':[],'reco_id':[],'ZERO':[],'begt':[],'endt':[],'NA1':[],'NA2':[],'speaker':[],'NA3':[],'NA4':[]}
	def time2sec(tim):
		tim = tim.split('=')[-1]
		if len(tim.split(':')) == 2:	
			sec = float(tim.split(':')[-1])
			minute = int(tim.split(':')[-2])
			sec = sec + 60 * minute
		elif len(tim.split(':')) == 3:
			sec = float(tim.split(':')[-1])
			minute = int(tim.split(':')[-2])
			hour = int(tim.split(':')[-3])
			sec = sec + 60 * minute + 3600 * hour
		else:
			raise ValueError('The time format is wrong!')
		sec = round(sec,2)
		return sec

	for lbe_file in os.listdir(lbe_dir):
		lbe_name = lbe_file.split('.')[0]
		lbe_path = os.path.join(lbe_dir,lbe_file)
		f = open(lbe_path,"rb")
		lbe_lines = f.readlines()
		f.close()
		pd_info['PREFIX'] += list(map(lambda x:"SPEAKER",lbe_lines[1:]))
		pd_info['ZERO'] += list(map(lambda x:"0",lbe_lines[1:]))
		pd_info['NA1'] += list(map(lambda x:"<NA>",lbe_lines[1:]))
		pd_info['NA2'] += list(map(lambda x:"<NA>",lbe_lines[1:]))
		pd_info['NA3'] += list(map(lambda x:"<NA>",lbe_lines[1:]))
		pd_info['NA4'] += list(map(lambda x:"<NA>",lbe_lines[1:]))
		pd_info['begt'] += list(map(lambda x:round(time2sec(x.split(b' ')[0].decode('UTF-8')),2),lbe_lines[1:]))
		pd_info['endt'] += list(map(lambda x:round(time2sec(x.split(b' ')[1].decode('UTF-8'))-time2sec(x.split(b' ')[0].decode('UTF-8')),2),lbe_lines[1:]))
		pd_info['reco_id'] += list(map(lambda x:lbe_name,lbe_lines[1:]))
		pd_info['speaker'] += list(map(lambda x:"%s"%(x[1].split(b' ')[3].decode('UTF-8').split('=')[1]),enumerate(lbe_lines[1:])))
	df = pd.DataFrame(pd_info)
	df = df.sort_values(['reco_id','begt'],ascending=True)
	df.to_csv(os.path.join(output_dir,'ref.rttm'),sep=' ',index=False,header=False)
def write_wav_scp(wav_dir,output_dir):
	pd_info = {'file_name':[],'path':[]}
	for wav_file in os.listdir(wav_dir):
		if wav_file.split('.')[-1] == 'wav':
			pd_info['file_name'].append(wav_file.split('.')[0])
			pd_info['path'].append(os.path.join(wav_dir,wav_file))
	df = pd.DataFrame(pd_info)
	df = df.sort_values('file_name',ascending=True)
	df.to_csv(os.path.join(output_dir,'wav.scp'),sep=' ',index=False,header=False)
def write_utt2spk(lbe_dir,output_dir):
	pd_info = {'utt_id':[],'spk_id':[]}
	for lbe_file in os.listdir(lbe_dir):
		lbe_name = lbe_file.split('.')[0]
		lbe_path = os.path.join(lbe_dir,lbe_file)
		f = open(lbe_path,"rb")
		lbe_lines = f.readlines()
		f.close()
		pd_info['spk_id'] += list(map(lambda x:"%s"%(lbe_name),lbe_lines[1:]))
		pd_info['utt_id'] += list(map(lambda x:"%s_%s"%(lbe_name,(4-len(str(x[0])))*"0"+str(x[0])),enumerate(lbe_lines[1:])))

		# pd_info['spk_id'] += list(map(lambda x:"%s-%s"%(lbe_name,x.split(b' ')[3].decode('UTF-8').split('=')[1]),lbe_lines[1:]))
		# pd_info['utt_id'] += list(map(lambda x:"%s-%s-%d"%(lbe_name,x[1].split(b' ')[3].decode('UTF-8').split('=')[1],x[0]),enumerate(lbe_lines[1:])))

	df = pd.DataFrame(pd_info)
	df = df.sort_values('utt_id',ascending=True)
	df.to_csv(os.path.join(output_dir,'utt2spk'),sep=' ',index=False,header=False)

def write_segments(lbe_dir,output_dir):
	pd_info = {'utt_id':[],'reco_id':[],'begt':[],'endt':[]}
	def time2sec(tim):
		tim = tim.split('=')[-1]
		if len(tim.split(':')) == 2:	
			sec = float(tim.split(':')[-1])
			minute = int(tim.split(':')[-2])
			sec = sec + 60 * minute
		elif len(tim.split(':')) == 3:
			sec = float(tim.split(':')[-1])
			minute = int(tim.split(':')[-2])
			hour = int(tim.split(':')[-3])
			sec = sec + 60 * minute + 3600 * hour
		else:
			raise ValueError('The time format is wrong!')
		sec = round(sec,2)
		return sec

	for lbe_file in os.listdir(lbe_dir):
		lbe_name = lbe_file.split('.')[0]
		lbe_path = os.path.join(lbe_dir,lbe_file)

		f = open(lbe_path,"rb")
		lbe_lines = f.readlines()
		f.close()
		pd_info['begt'] += list(map(lambda x:time2sec(x.split(b' ')[0].decode('UTF-8')),lbe_lines[1:]))
		pd_info['endt'] += list(map(lambda x:time2sec(x.split(b' ')[1].decode('UTF-8')),lbe_lines[1:]))
		pd_info['reco_id'] += list(map(lambda x:lbe_name,lbe_lines[1:]))
		pd_info['utt_id'] += list(map(lambda x:"%s_%s"%(lbe_name,(4-len(str(x[0])))*"0"+str(x[0])),enumerate(lbe_lines[1:])))
	df = pd.DataFrame(pd_info)
	df = df.sort_values('utt_id',ascending=True)
	df.to_csv(os.path.join(output_dir,'segments'),sep=' ',index=False,header=False)

def main():
	rawdata_path=sys.argv[1]
	scp_path=sys.argv[2]
	lbe_dir = rawdata_path+"/lbe"
	wav_dir = rawdata_path+"/wav"
	output_dir = scp_path+'/Hub4m97'
	output_dir1 = scp_path+'/Hub4m97_1'
	output_dir2 = scp_path+'/Hub4m97_2'
	# vad_dir = "mfcc"
	# mfcc_dir = "mfcc"
	# mfcclog_dir = "exp/make_mfcc"
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	if not os.path.exists(output_dir1):
		os.mkdir(output_dir1)
	if not os.path.exists(output_dir2):
		os.mkdir(output_dir2)
	# if not os.path.exists(mfcc_dir):
	# 	os.mkdir(mfcc_dir)
	# if not os.path.exists(mfcclog_dir):
	# 	os.mkdir(mfcclog_dir)
	# if not os.path.exists(vad_dir):
	# 	os.mkdir(vad_dir)
	
	# Hub4m97
	write_wav_scp(wav_dir,output_dir)
	write_segments(lbe_dir,output_dir)
	write_utt2spk(lbe_dir,output_dir)
	write_rttm(lbe_dir,output_dir)
	write_reco2num_spk(lbe_dir,output_dir)
	os.system('utils/utt2spk_to_spk2utt.pl %s > %s'%(os.path.join(output_dir,'utt2spk'),os.path.join(output_dir,'spk2utt')))


if __name__ == "__main__":
	main()

