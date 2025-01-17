#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data
res_path=$exp_path/results/word2vec_perturb
mkdir -p $exp_path $data_path $res_path

datasets="writing xsum squad"
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B gpt-j-6B gpt-neox-20b"


# # augmenting dataset
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Preparing dataset ${D}_${M} ...
#     python3 scripts/word2vec_perturb/data_builder.py --dataset $D --n_samples 500 --base_model_name $M --output_file $data_path/${D}_${M}
#   done
# done

#White-box Setting
echo `date`, Evaluate models in the white-box setting:

# evaluate Fast-DetectGPT and fast baselines
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating Fast-DetectGPT on ${D}_${M} ...
    python3 scripts/word2vec_perturb/fast_detect_gpt.py --reference_model_name $M --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/white/fast_detect/${D}_${M}

    echo `date`, Evaluating baseline methods on ${D}_${M} ...
    python3 scripts/word2vec_perturb/baselines.py --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/white/baseline/${D}_${M}
  done
done