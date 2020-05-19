#!/bin/bash
# Copyright 2017-2018  David Snyder
#           2017-2018  Matthew Maciejewski
#
# Apache 2.0.
#
# This recipe demonstrates the use of x-vectors for speaker diarization.
# The scripts are based on the recipe in ../v1/run.sh, but clusters x-vectors
# instead of i-vectors.  It is similar to the x-vector-based diarization system
# described in "Diarization is Hard: Some Experiences and Lessons Learned for
# the JHU Team in the Inaugural DIHARD Challenge" by Sell et al.  The main
# difference is that we haven't implemented the VB resegmentation yet.

. ./cmd.sh 
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
Hub4m97_root=/mnt/nas_workspace2/spmiData/Hub4m97
Deliver_root=/mnt/nas_data/Deliver2Sinovoice_2014Nov/data_20140311_third
stage=5
nnet_dir=exp/xvector_nnet_1a/

JB_path=exp/JB 
JB_para_path=$JB_path/JB_parameter
dev_dataset=Hub4m97_1
test_dataset=Hub4m97_2
dataset=Hub4m97
dev_path=$JB_path/xvectors_${dev_dataset}_dev
test_path=$JB_path/xvectors_${test_dataset}_test
mkdir -p $dev_path
mkdir -p $test_path
# Prepare datasets
if [ $stage -le 0 ]; then
  if [ $dataset -eq Hub4m97 ]; then
    # 生成kaldi格式的Hub4m97数据集,并将该数据集分割成recording相等的两部分.每一个部分有29个recording.
    echo "0 Preparing Hub4m97..."
    python dataloader/make_Hub4m97.py $Hub4m97_root data
    steps/make_mfcc.sh --mfcc-config conf/mfcc_Hub4m97.conf --nj 16 \
        --cmd "$train_cmd" --write-utt2num-frames true \
        data/Hub4m97 exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/Hub4m97

    sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
        data/Hub4m97 exp/make_vad $vaddir
    utils/fix_data_dir.sh data/Hub4m97
    # 将Hub4m97分割为Hub4m97_1和Hub4m97_2
    python utils_Hub4m97/split_Hub4m97.py data
  fi
fi

# Prepare features
if [ $stage -le 1 ]; then
  echo "1 making mfcc on Hub4m97 dataset..."
  for name in $dev_dataset $test_dataset; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 16 \
        --cmd "$train_cmd" --write-utt2num-frames true \
        data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}

    sid/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
        data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}

    local/nnet3/xvector/prepare_feats.sh --nj 16 --cmd "$train_cmd" \
        data/${name} data/${name}_cmn exp/${name}_cmn
    cp data/${name}/vad.scp data/${name}_cmn/
    if [ -f data/${name}/segments ]; then
        cp data/${name}/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done
fi

# Extract x-vectors
if [ $stage -le 2 ]; then

  echo "2 extract x-vectors on dev and test dataset..."
  # Extract x-vectors for the two partitions of callhome.
  for name in $dev_dataset $test_dataset; do
    diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
      --nj 20 --window 1.5 --period 0.75 --apply-cmn false \
      --min-segment 0.5 $nnet_dir \
      data/${name}_cmn $nnet_dir/xvectors_${name}
  done
fi

# whitening
if [ $stage -le 3 ]; then
  echo "3 scaling dev and test dataset..."
####################################################
  mkdir -p $dev_path
  mkdir -p $test_path
  
  #开发集
  ivector-subtract-global-mean $nnet_dir/xvectors_${dev_dataset}/mean.vec scp:$nnet_dir/xvectors_${dev_dataset}/xvector.scp ark:- | \
  ivector-normalize-length ark:- ark,scp:$dev_path/xvector.ark,$dev_path/xvector.scp


  mkdir -p $JB_path/log
  $train_cmd $JB_path/log/transform.log \
  est-pca --read-vectors=true --normalize-mean=false \
    --normalize-variance=true --dim=-1 \
    scp:$dev_path/xvector.scp $dev_path/transform.mat || exit 1;


  #开发集变换
  ivector-subtract-global-mean $nnet_dir/xvectors_${dev_dataset}/mean.vec scp:$nnet_dir/xvectors_${dev_dataset}/xvector.scp ark:- | \
  ivector-normalize-length ark:- ark:- | \
  transform-vec $dev_path/transform.mat ark:- ark,scp:$dev_path/xvector.ark,$dev_path/xvector.scp  
  # ivector-normalize-length ark:- ark,scp:$dev_path/xvector.ark,$dev_path/xvector.scp  

  #### 对测试集进行去均值,白化以及长度归一化 ####

  ivector-subtract-global-mean $nnet_dir/xvectors_${dev_dataset}/mean.vec scp:$nnet_dir/xvectors_${test_dataset}/xvector.scp ark:- | \
  ivector-normalize-length ark:- ark:- | \
  transform-vec $dev_path/transform.mat ark:- ark,scp:$test_path/xvector.ark,$test_path/xvector.scp
  # ivector-normalize-length ark:- ark,scp:$test_path/xvector.ark,$test_path/xvector.scp

  # #开发集变换
  # ivector-subtract-global-mean $nnet_dir/xvectors_${dev_dataset}/mean.vec scp:$nnet_dir/xvectors_${dev_dataset}/xvector.scp ark:- | \
  # ivector-normalize-length ark:- ark,scp:$dev_path/xvector.ark,$dev_path/xvector.scp


  # #### 对测试集进行去均值,白化以及长度归一化 ####

  # ivector-subtract-global-mean $nnet_dir/xvectors_${dev_dataset}/mean.vec scp:$nnet_dir/xvectors_${test_dataset}/xvector.scp ark:- | \
  # ivector-normalize-length ark:- ark,scp:$test_path/xvector.ark,$test_path/xvector.scp


  cp $nnet_dir/xvectors_${test_dataset}/utt2spk $test_path/utt2spk 
  cp $nnet_dir/xvectors_${test_dataset}/spk2utt $test_path/spk2utt 
  cp $nnet_dir/xvectors_${test_dataset}/segments $test_path/segments 



  cp $nnet_dir/xvectors_${dev_dataset}/utt2spk $dev_path/utt2spk 
  cp $nnet_dir/xvectors_${dev_dataset}/spk2utt $dev_path/spk2utt 
  cp $nnet_dir/xvectors_${dev_dataset}/segments $dev_path/segments 
fi

# calculate score
if [ $stage -le 4 ]; then
  echo "4 calculating score on test dataset..."
  python JB/test.py $test_path $JB_para_path $dev_path/transform.mat $test_path
  python JB/test.py $dev_path $JB_para_path $dev_path/transform.mat $dev_path
fi


# Cluster the JB scores using a stopping threshold.
if [ $stage -le 5 ]; then
  # First, we find the threshold that minimizes the DER on each partition of
  # callhome.
    mkdir -p $JB_path/tuning
    echo "Tuning clustering threshold for $dev_dataset"
    best_der=100
    best_threshold=0
    utils/filter_scp.pl -f 2 data/$dev_dataset/wav.scp \
        data/$dataset/ref.rttm > data/$dev_dataset/ref.rttm
    #for threshold in -1.7 -1.8 -1.9 -2 -2.1 -2.2 -2.3 -2.4; do
    for threshold in 1.3 1.2 1.1 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
        diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
            --threshold $threshold $dev_path \
            $dev_path/JB_scores_t$threshold
        
        md-eval.pl -1 -c 0.25 -r data/$dev_dataset/ref.rttm \
            -s $dev_path/JB_scores_t$threshold/rttm \
            2> $JB_path/tuning/${dev_dataset}_t${threshold}.log \
            > $JB_path/tuning/${dev_dataset}_t${threshold}

        der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
            ${JB_path}/tuning/${dev_dataset}_t${threshold})
        echo "$der"
        if [ $(echo $der'<'$best_der | bc -l) -eq 1 ]; then
            best_der=$der
            best_threshold=$threshold
            
        fi
        done
    echo "$best_threshold" > $JB_path/tuning/${dev_dataset}_best
    echo "the best der is ${best_der}"
fi


if [ $stage -le 6 ]; then
  # Cluster test_dataset using the best threshold found for dev_dataset.  This way,
  # dev_dataset is treated as a held-out dataset to discover a reasonable
  # stopping threshold for test_dataset.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat exp/JB/tuning/${dev_dataset}_best) \
    $test_path $test_path/JB_scores


  mkdir -p $JB_path/results
  # Now combine the results for Hub4m97_1 and Hub4m97_2 and evaluate it
  # together.
  
  cat $test_path/JB_scores/rttm | md-eval.pl -1 -c 0.25 -r \
    data/$test_dataset/ref.rttm -s - 2> $JB_path/results/threshold.log \
    > $JB_path/results/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $JB_path/results/DER_threshold.txt)
  # Using supervised calibration, DER: 8.39%
  # Compare to 10.36% in ../v1/run.sh
  echo "Using supervised calibration, DER: $der%"
fi
# Cluster the JB scores using the oracle number of speakers
if [ $stage -le 7 ]; then
  # diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
  #   --reco2num-spk data/$dev_dataset/reco2num_spk \
  #   $dev_path $dev_path/JB_scores_num_spk

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/$test_dataset/reco2num_spk \
    $test_path $test_path/JB_scores_num_spk

  mkdir -p $JB_path/results
  # Now combine the results for Hub4m97_1 and Hub4m97_2 and evaluate it together.
  cat $test_path/JB_scores_num_spk/rttm \
    | md-eval.pl -1 -c 0.25 -r data/$test_dataset/ref.rttm -s - 2> $JB_path/results/num_spk.log \
    > $JB_path/results/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $JB_path/results/DER_num_spk.txt)
  echo "Using the oracle number of speakers, DER: $der%"
fi

# total evaluations
if [ $stage -le 8 ]; then
  cat $JB_path/xvectors_${dev_dataset}_test/JB_scores/rttm \
    $JB_path/xvectors_${test_dataset}_test/JB_scores/rttm | md-eval.pl -1 -c 0.25 -r \
    data/$dataset/ref.rttm -s - 2> $JB_path/results/threshold.log \
    > $JB_path/results/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/JB/results/DER_threshold.txt)
  # Using supervised calibration, DER: 8.39%
  # Compare to 10.36% in ../v1/run.sh
  echo "Using supervised calibration, DER: $der%"

  cat $JB_path/xvectors_${dev_dataset}_test/JB_scores_num_spk/rttm \
  $JB_path/xvectors_${test_dataset}_test/JB_scores_num_spk/rttm \
    | md-eval.pl -1 -c 0.25 -r data/$dataset/ref.rttm -s - 2> $JB_path/results/num_spk.log \
    > $JB_path/results/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/JB/results/DER_num_spk.txt)
  echo "Using the oracle number of speakers, DER: $der%"
fi
