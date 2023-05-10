export CUDA_VISIBLE_DEVICES=0
# export LOGLEVEL=DEBUG

# my_domain="medical_low"
domain="law"
# datastore_domain="wiki"
num_neighbors=8 #D top
lambda=0.7
temperature=5

knn_models=../.././knn_models
multi_domin_corpus=./multi_domain_new_split
data_bin=./multi-domain-data-bin/${domain}
datastore=./multi-domain-datastore/${domain}
datastore_size=18857646
# total_tokens=6916846
checkpoint=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/pretrained_model/wmt19.de-en.ffn8192.pt
output_path=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/result
auto_encoder_path=/users10/lhuang/xiaokenaifan/ConAE/train/MSMARCO/multi-domain-data-datastore/medical/save_model_linears1_64_nei2/model.best.pt
# pos_sample=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/multi-domain-datastore
# neg_sample=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/multi-domain-datastore
max_tokens=8000

# python /users10/lhuang/miniconda3/envs/knn/lib/python3.8/site-packages/fairseq_cli/generate.py ${data_bin} \
fairseq-generate ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_knn \
    --datastore ${datastore} \
    --keys-dimension 64 \
    --input-dim 1024 \
    --output-dim 64 \
    --auto-encoder ${auto_encoder_path} \
    --datastore-size ${datastore_size} \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
    --lambda-value ${lambda} \
    --temperature-value ${temperature} \
    --source-lang de \
    --target-lang en \
    --gen-subset test \
    --path ${checkpoint} \
    --max-tokens ${max_tokens} \
    --beam 5 \
    --tokenizer moses \
    --post-process subword_nmt \
    > ${output_path}/medical_sys.de-en.en

cat ${output_path}/medical_sys.de-en.en | grep -P "^D" | sort -V | cut -f 3- > sys.de-en.en
sacrebleu -w 6 ${multi_domin_corpus}/${domain}/test.en --input sys.de-en.en