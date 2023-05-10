export CUDA_VISIBLE_DEVICES=0

domain="medical"
# datastore_domain="wiki"
knn_models=../.././knn_models
multi_domin_corpus=./multi_domain_new_split
data_bin=./multi-domain-data-bin/${domain}
datastore=./multi-domain-datastore-train/${domain}
datastore_size=6501418
checkpoint=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/pretrained_model/wmt19.de-en.ffn8192.pt
auto_encoder_path=/users10/lhuang/xiaokenaifan/ConAE/train/MSMARCO/multi-domain-data-datastore/medical/save_model_64/model.best.pt
root=/users10/lhuang/xiaokenaifan/knn-models/knn_models_cli
max_tokens=8000

python ${root}/tune_knn_params.py \
    --reference ${multi_domin_corpus}/${domain}/test.en \
    --candidate-num-neighbors 2 4 8 16\
    --candidate-lambda-value 0.1 0.2 0.3 0.4 0.5 0.6 0.7\
    --candidate-temperature-value 1 5 10\
    --sacrebleu-args "-w 6" \
    $(which fairseq-generate) ${data_bin} \
        --user-dir ${knn_models} \
        --task translation_knn \
        --datastore ${datastore} \
        --keys-dimension 64 \
        --input-dim 1024 \
        --output-dim 64 \
        --auto-encoder ${auto_encoder_path} \
        --datastore-size ${datastore_size} \
        --knn-fp16 \
        --source-lang de \
        --target-lang en \
        --gen-subset test \
        --path ${checkpoint} \
        --max-tokens ${max_tokens} \
        --beam 5 \
        --tokenizer moses \
        --post-process subword_nmt >/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/log/ae_medical_tune_test.log 2>&1 &