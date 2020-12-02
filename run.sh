. cmd.sh
. path.sh

train_cmd_intel=run.pl
name=train
nnet_dir=exp/xvector_nnet_1a
mfccdir=mfcc
nj=$(nproc)

# Make features
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
--cmd "$train_cmd_intel" --write-utt2num-frames true \
data/$name exp/make_mfcc $mfccdir
utils/fix_data_dir.sh data/$name
local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$train_cmd_intel" \
data/$name data/${name}_cmn exp/${name}_cmn

# Make segments
# Create vad
# steps/compute_vad_decision.sh --nj $nj --vad-config conf/vad.conf --cmd "$train_cmd_intel" \
# data/$name exp/make_vad $mfccdir
# Create segments
# diarization/vad_to_segments.sh --nj $nj --cmd "$train_cmd_intel" data/$name data/${name}_segmented

# Copy segments
cp data/$name/segments data/${name}_cmn/
utils/fix_data_dir.sh data/${name}_cmn

# Extract embeddings
diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd_intel" \
--nj $nj --window 1.5 --period 0.75 --apply-cmn false \
--min-segment 0.5 $nnet_dir \
data/${name}_cmn $nnet_dir/xvectors_${name}

# Perform PLDA scoring
diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd_intel" \
--target-energy 0.9 --nj $nj $nnet_dir/xvectors_plda/ \
$nnet_dir/xvectors_$name \
$nnet_dir/xvectors_$name/plda_scores

# Cluster Speakers
diarization/cluster.sh --cmd "$train_cmd_intel" --nj $nj \
--reco2num-spk data/$name/reco2num_spk \
$nnet_dir/xvectors_$name/plda_scores \
$nnet_dir/xvectors_$name/plda_scores_num_speakers

# threshold=0.5
# diarization/cluster.sh --cmd "$train_cmd_intel --mem 4G" --nj 40 \
# --threshold $threshold \
# $nnet_dir/xvectors_$name/plda_scores \
# $nnet_dir/xvectors_$name/plda_scores_threshold_${threshold}
