# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data/hybrid
res_path=$exp_path/results/hybrid
mkdir -p $exp_path $data_path $res_path

datasets="writing xsum squad"
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B gpt-j-6B gpt-neox-20b"
# embedding_source="glove fasttext word2vec tmae elmo bert"
embedding_source="glove fasttext word2vec tmae bert"


# perturb dataset
# for D in $datasets; do
#   for M in $source_models; do
#     for E in $embedding_source; do
#       echo `date`, Preparing dataset ${D}_${M}_${E} ...
#       python3 scripts/hybrid_perturb/perturb_generator.py --dataset $D --n_samples 500 --base_model_name $M --embedding $E --output_file $data_path/${D}_${M}
#     done
#   done
# done

for E in $embedding_source; do
  #White-box Setting
  echo `date`, Evaluate models in the white-box setting:

  # evaluate Fast-DetectGPT and fast baselines
  for D in $datasets; do
    for M in $source_models; do
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M} ...
      python3 scripts/hybrid_perturb/fast_detect_gpt.py --reference_model_name $M --scoring_model_name $M --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/white/fast_detect/${D}_${M}  --embedding $E

      echo `date`, Evaluating baseline methods on ${D}_${M} ...
      python3 scripts/hybrid_perturb/baselines.py --scoring_model_name $M --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/white/baseline/${D}_${M} --embedding $E
    done
  done

  # evaluate DNA-GPT
  for D in $datasets; do
    for M in $source_models; do
      echo `date`, Evaluating DNA-GPT on ${D}_${M} ...
      python3 scripts/hybrid_perturb/dna_gpt.py --base_model_name $M --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/white/dna_gpt/${D}_${M} --embedding $E
    done
  done

  # evaluate DetectGPT and its improvement DetectLLM
  for D in $datasets; do
    for M in $source_models; do
      echo `date`, Evaluating DetectGPT on ${D}_${M} ...
      python3 scripts/hybrid_perturb/detect_gpt.py --scoring_model_name $M --mask_filling_model_name t5-3b --n_perturbations 100 --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/white/detect_gpt/${D}_${M} --embedding $E
      # we leverage DetectGPT to generate the perturbations
      echo `date`, Evaluating DetectLLM methods on ${D}_${M} ...
      python3 scripts/hybrid_perturb/detect_llm.py --scoring_model_name $M --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/white/detect_llm/${D}_${M} --embedding $E
    done
  done

  # Black-box Setting
  echo `date`, Evaluate models in the black-box setting:
  scoring_models="gpt-neo-2.7B"

  # evaluate Fast-DetectGPT
  for D in $datasets; do
    for M in $source_models; do
      M1=gpt-j-6B  # sampling model
      for M2 in $scoring_models; do
        echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
        python3 scripts/hybrid_perturb/fast_detect_gpt.py --reference_model_name ${M1} --scoring_model_name ${M2} --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/black/fast_detect/${D}_${M}.${M1}_${M2} --embedding $E
      done
    done
  done

  # evaluate DetectGPT and its improvement DetectLLM
  for D in $datasets; do
    for M in $source_models; do
      M1=t5-3b  # perturbation model
      for M2 in $scoring_models; do
        echo `date`, Evaluating DetectGPT on ${D}_${M}.${M1}_${M2} ...
        python3 scripts/hybrid_perturb/detect_gpt.py --mask_filling_model_name ${M1} --scoring_model_name ${M2} --n_perturbations 100 --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/black/detect_gpt/${D}_${M}.${M1}_${M2} --embedding $E
        # we leverage DetectGPT to generate the perturbations
        echo `date`, Evaluating DetectLLM methods on ${D}_${M}.${M1}_${M2} ...
        python3 scripts/hybrid_perturb/detect_llm.py --scoring_model_name ${M2} --dataset $D \
                            --dataset_file $data_path/${D}_${M} --output_file $res_path/black/detect_llm/${D}_${M}.${M1}_${M2} --embedding $E
      done
    done
  done
done