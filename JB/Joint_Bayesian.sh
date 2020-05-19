#!/usr/bin/env bash
stage=0
path_train=$1
train_output=$2
test_dataset=$3
test_output=$4
iter_count=100
if [ $stage -le 0 ]; then
    echo "training Joint Bayesian..."
    python JB/train.py "$path_train/xvector.scp" "$path_train/utt2spk" $train_output $iter_count
    ############## the below line is debugging
    # aaa=/exp/joint_bayesian/sre_combined
    # bbb=data/sre_combined
    # train_outputccc=/mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v2/exp/jb/xvectors_callhome1/JB
    # iter_count=10
    # python /mnt/workspace2/yuly/kaldi/egs/callhome_diarization/v2/JB/train.py "$aaa/xvector.scp" "$bbb/utt2spk" $train_outputccc $iter_count
    # ############### the above line is debugging
fi

if [ $stage -le 1 ]; then
    echo "calculating scores..."
    cov_path=$train_output
    #the below line is debugging.
    # cov_path=/mnt/workspace1/liyt/kaldi/egs/sre18/v2/exp/joint_bayesian/model
    echo "calculating dataset ${test_dataset}"
    echo "python JB/test.py "exp/jb/xvectors_${test_dataset}" $cov_path $test_output"
    python JB/test.py "exp/jb/xvectors_${test_dataset}" $cov_path $test_output
fi

