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
sre_root=/mnt/nas_data/sredata/LDC2009E100_sre96_08
swbd_root=/mnt/nas_data/sredata/LDC2016E81_SRE16_SWB/LDC2016E81
musan_root=/mnt/nas_workspace2/spmiData/musan
stage=4
nnet_dir=exp/xvector_nnet_1a/

num_components=1024 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)
# Prepare datasets
if [ $stage -le 0 ]; then
  # Prepare a collection of NIST SRE data. This will be used to train,
  # x-vector DNN and PLDA model.
  echo "0.1 Preparing sre..."
  local/make_sre.sh $sre_root data

  # Prepare SWB for x-vector DNN training.
  echo "0.2 Preparing swbd..."
  local/make_swbd2_phase1.pl $swbd_root/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl $swbd_root/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl $swbd_root/LDC2002S06 \
                           data/swbd2_phase3_train
  local/make_swbd_cellular1.pl $swbd_root/LDC2001S13/swb_cell_1_audio_d1 \
                             data/swbd_cellular1_train1
  local/make_swbd_cellular1.pl $swbd_root/LDC2001S13/swb_cell_1_audio_d2 \
                             data/swbd_cellular1_train2
  
  local/make_swbd_cellular2.pl $swbd_root/LDC2004S07/swbcell2_01 \
                             data/swbd_cellular2_train1
  local/make_swbd_cellular2.pl $swbd_root/LDC2004S07/swbcell2_02 \
                             data/swbd_cellular2_train2
  local/make_swbd_cellular2.pl $swbd_root/LDC2004S07/swbcell2_03 \
                             data/swbd_cellular2_train3



  # Combing all training data
  echo "0.4 Combing all training data..."
  utils/combine_data.sh data/train \
    data/swbd_cellular1_train1 data/swbd_cellular1_train2 \
    data/swbd_cellular2_train1 data/swbd_cellular2_train2 data/swbd_cellular2_train3 \
    data/swbd2_phase1_train \
    data/swbd2_phase2_train data/swbd2_phase3_train data/sre
fi

# Prepare features
if [ $stage -le 1 ]; then
  # The script local/make_callhome.sh splits callhome into two parts, called
  # callhome1 and callhome2.  Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  echo "stage1.1: making mfcc on training dataset..."
  for name in train; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  for name in train; do
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  done

fi

# In this section, we augment the training data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh $musan_root data
  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  utils/subset_data_dir.sh data/train_aug 128000 data/train_aug_128k
  utils/fix_data_dir.sh data/train_aug_128k

  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/train_aug_128k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_128k data/train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/train_combined data/train_combined_cmn_no_sil exp/train_combined_cmn_no_sil
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/train_combined_cmn_no_sil/utt2num_frames data/train_combined_cmn_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_cmn_no_sil/utt2num_frames.bak > data/train_combined_cmn_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_combined_cmn_no_sil/utt2num_frames data/train_combined_cmn_no_sil/utt2spk > data/train_combined_cmn_no_sil/utt2spk.new
  mv data/train_combined_cmn_no_sil/utt2spk.new data/train_combined_cmn_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_combined_cmn_no_sil/spk2utt > data/train_combined_cmn_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' \
    data/train_combined_cmn_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_cmn_no_sil/spk2utt \
    > data/train_combined_cmn_no_sil/spk2utt.new
  mv data/train_combined_cmn_no_sil/spk2utt.new data/train_combined_cmn_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_combined_cmn_no_sil/spk2utt > data/train_combined_cmn_no_sil/utt2spk

  utils/filter_scp.pl data/train_combined_cmn_no_sil/utt2spk data/train_combined_cmn_no_sil/utt2num_frames > data/train_combined_cmn_no_sil/utt2num_frames.new
  mv data/train_combined_cmn_no_sil/utt2num_frames.new data/train_combined_cmn_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil
fi

local/nnet3/xvector/tuning/run_xvector_1a.sh --stage $stage --train-stage -1 \
  --data data/train_combined_cmn_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs
