# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data/hybrid_perturb
res_path=$exp_path/results/hybrid_perturb
mkdir -p $exp_path $data_path $res_path

datasets="writing xsum squad"
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B gpt-j-6B gpt-neox-20b"
# embedding_source="glove fasttext word2vec tmae elmo bert"
embedding_source="word2vec"


# augmenting dataset
for D in $datasets; do
  for M in $source_models; do
    for E in $embedding_source; do
      echo `date`, Preparing dataset ${D}_${M}_${E} ...
      python3 scripts/hybrid_perturb/perturb_generator.py --dataset $D --n_samples 500 --base_model_name $M --embedding $E --output_file $data_path/${D}_${M}
    done
  done
done

#White-box Setting
# echo `date`, Evaluate models in the white-box setting:

# # evaluate Fast-DetectGPT and fast baselines
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating Fast-DetectGPT on ${D}_${M} ...
#     python3 scripts/word2vec_perturb/fast_detect_gpt.py --reference_model_name $M --scoring_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/white/fast_detect/${D}_${M}

#     # echo `date`, Evaluating baseline methods on ${D}_${M} ...
#     # python3 scripts/word2vec_perturb/baselines.py --scoring_model_name $M --dataset $D \
#     #                       --dataset_file $data_path/${D}_${M} --output_file $res_path/white/baseline/${D}_${M}
#   done
# done

# # Black-box Setting
# echo `date`, Evaluate models in the black-box setting:
# scoring_models="gpt-neo-2.7B"

# # evaluate Fast-DetectGPT
# for D in $datasets; do
#   for M in $source_models; do
#     M1=gpt-j-6B  # sampling model
#     for M2 in $scoring_models; do
#       echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
#       python3 scripts/word2vec_perturb/fast_detect_gpt.py --reference_model_name ${M1} --scoring_model_name ${M2} --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/black/fast_detect/${D}_${M}.${M1}_${M2}
#     done
#   done
# done