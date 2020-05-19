import os
global utt_id
from scipy.io import wavfile
import pandas as pd
import sys
sys.path.append("/mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v2/kaldi_io")
import kaldi_io
import numpy as np



def write_reco2num_spk(file_list,output_path):
	pd_info = {'reco_id':[],'num_spk':[]}
	for file_item in file_list:
		pd_info['reco_id'].append(file_item)
		pd_info['num_spk'].append(2)
	df = pd.DataFrame(pd_info)
	df = df.sort_values('reco_id',ascending=True)
	df.to_csv(os.path.join(output_path,'reco2num_spk'),sep=' ',index=False,header=False)

def write_rttm(rawdata_path,file_list,output_path):
    pd_info = {'PREFIX':[],'reco_id':[],'ZERO':[],'begt':[],'endt':[],'NA1':[],'NA2':[],'speaker':[],'NA3':[],'NA4':[]}
    def time2sec(tim):
        tim = tim.strip()
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
    
    for file_item in file_list:
        lbe_item = os.path.join(rawdata_path,file_item+'_1.lbe')
        f = open(lbe_item,"r")
        lbe_lines = f.readlines()
        f.close()
        #fix lbe_lines:
        i=1
        for item in lbe_lines[1:]:
            if item.split(' ')[2].split('=')[1]=='c':
                fix_item1 = item.strip()[:-1]+'a'
                fix_item2 = item.strip()[:-1]+'b'
                del lbe_lines[i]
                lbe_lines.insert(i,fix_item1)
                lbe_lines.insert(i,fix_item2)
                i += 1
            elif item.split(' ')[2].split('=')[1]=='ax':
                fix_item = item.strip()[:-2]+'a'
                del lbe_lines[i]
                lbe_lines.insert(i,fix_item)
            elif item.split(' ')[2].split('=')[1]=='bx':
                fix_item = item.strip()[:-2]+'b'
                del lbe_lines[i]
                lbe_lines.insert(i,fix_item)
            i += 1
        pd_info['PREFIX'] += list(map(lambda x:"SPEAKER",lbe_lines[1:]))
        pd_info['ZERO'] += list(map(lambda x:"0",lbe_lines[1:]))
        pd_info['NA1'] += list(map(lambda x:"<NA>",lbe_lines[1:]))
        pd_info['NA2'] += list(map(lambda x:"<NA>",lbe_lines[1:]))
        pd_info['NA3'] += list(map(lambda x:"<NA>",lbe_lines[1:]))
        pd_info['NA4'] += list(map(lambda x:"<NA>",lbe_lines[1:]))
        pd_info['begt'] += list(map(lambda x:round(time2sec(x.split(' ')[0]),2),lbe_lines[1:]))
        pd_info['endt'] += list(map(lambda x:round(time2sec(x.split(' ')[1])-time2sec(x.split(' ')[0]),2),lbe_lines[1:]))
        pd_info['reco_id'] += list(map(lambda x:file_item,lbe_lines[1:]))
        pd_info['speaker'] += list(map(lambda x:"%s"%(x[1].split(' ')[2].split('=')[1]),enumerate(lbe_lines[1:])))
    df = pd.DataFrame(pd_info)
    df = df.sort_values(['reco_id','begt'],ascending=True)
    df.to_csv(os.path.join(output_path,'ref.rttm'),sep=' ',index=False,header=False)
def write_wav_scp(root_dir,file_list,output_dir):
    pd_info = {'file_name':[],'path':[]}
    for file_item in file_list:
            pd_info['file_name'].append(file_item)
            pd_info['path'].append(os.path.join(root_dir,file_item+'.wav'))
    df = pd.DataFrame(pd_info)
    df = df.sort_values('file_name',ascending=True)
    df.to_csv(os.path.join(output_dir,'wav.scp'),sep=' ',index=False,header=False)
    print("the total of wav is",len(file_list))
def write_utt2spk(rawdata_path,file_list,output_path):
    pd_info = {'utt_id':[],'spk_id':[]}
    for file_item in file_list:
        seg_item = os.path.join(rawdata_path,file_item+'_reseg.res')
        f = open(seg_item,"r")
        seg_lines = f.readlines()
        f.close()
        pd_info['spk_id'] += list(map(lambda x:"%s"%(file_item),seg_lines[1:]))
        pd_info['utt_id'] += list(map(lambda x:"%s_%s"%(file_item,(4-len(str(x[0])))*"0"+str(x[0])),enumerate(seg_lines[1:])))

    df = pd.DataFrame(pd_info)
    df = df.sort_values('utt_id',ascending=True)
    df.to_csv(os.path.join(output_path,'utt2spk'),sep=' ',index=False,header=False)
def write_resegments(rawdata_path,file_list,output_path):
    pd_info = {'utt_id':[],'reco_id':[],'begt':[],'endt':[]}
    def time2sec(tim):
        tim = tim.strip()
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
    total_utt=0
    for file_item in file_list:
        seg_item = os.path.join(rawdata_path,file_item+'_reseg.res')
        f = open(seg_item,"r")
        seg_lines = f.readlines()
        total_utt+=len(seg_lines[1:])

        f.close()
        pd_info['begt'] += list(map(lambda x:time2sec(x.split(' ')[0]),seg_lines[1:]))
        pd_info['endt'] += list(map(lambda x:time2sec(x.split(' ')[1]),seg_lines[1:]))
        pd_info['reco_id'] += list(map(lambda x:file_item,seg_lines[1:]))
        pd_info['utt_id'] += list(map(lambda x:"%s_%s"%(file_item,(4-len(str(x[0])))*"0"+str(x[0])),enumerate(seg_lines[1:])))
    df = pd.DataFrame(pd_info)
    df = df.sort_values('utt_id',ascending=True)
    df.to_csv(os.path.join(output_path,'segments'),sep=' ',index=False,header=False)
    print("the total of utterances is",total_utt)
def write_segments(rawdata_path,file_list,output_path):
    pd_info = {'utt_id':[],'reco_id':[],'begt':[],'endt':[]}
    def time2sec(tim):
        tim = tim.strip()
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
    total_utt=0
    for file_item in file_list:
        seg_item = os.path.join(rawdata_path,file_item+'_reseg.res')
        f = open(seg_item,"r")
        seg_lines = f.readlines()
        total_utt+=len(seg_lines[1:])
        f.close()
        pd_info['begt'] += list(map(lambda x:time2sec(x.split(' ')[0]),seg_lines[1:]))
        pd_info['endt'] += list(map(lambda x:time2sec(x.split(' ')[1]),seg_lines[1:]))
        pd_info['reco_id'] += list(map(lambda x:file_item,seg_lines[1:]))
        pd_info['utt_id'] += list(map(lambda x:"%s_%s"%(file_item,(4-len(str(x[0])))*"0"+str(x[0])),enumerate(seg_lines[1:])))
    df = pd.DataFrame(pd_info)
    df = df.sort_values('utt_id',ascending=True)
    df.to_csv(os.path.join(output_path,'segments'),sep=' ',index=False,header=False)
    print("the total of utterances is",total_utt)
def main():
    rawdata_path=sys.argv[1]
    output_path=os.path.join(sys.argv[2],'Deliver')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    file_list = os.listdir(os.path.join(rawdata_path,'filelist'))
    file_list = list(map(lambda x:x.split('.')[0],file_list))
    
    write_wav_scp(rawdata_path,file_list,output_path)
    #write_segments(rawdata_path,file_list,output_path)
    write_resegments(rawdata_path,file_list,output_path)
    write_utt2spk(rawdata_path,file_list,output_path)
    write_rttm(rawdata_path,file_list,output_path)
    write_reco2num_spk(file_list,output_path)
    os.system('utils/utt2spk_to_spk2utt.pl %s > %s'%(os.path.join(output_path,'utt2spk'),os.path.join(output_path,'spk2utt')))

if __name__ == "__main__":
	main()

