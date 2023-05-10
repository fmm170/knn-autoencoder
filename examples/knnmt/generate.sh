export CUDA_VISIBLE_DEVICES=0
domain="medical"
# datastore_domain="wiki"

# knn_models=../.././knn_models
multi_domin_corpus=./multi_domain_new_split
data_bin=./multi-domain-data-bin/${domain}

checkpoint=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/checkpoints/checkpoint_average.pt

max_tokens=8000

fairseq-generate ${data_bin} \
    --task translation \
    --source-lang de \
    --target-lang en \
    --gen-subset test \
    --path ${checkpoint} \
    --max-tokens ${max_tokens} \
    --beam 5 \
    --tokenizer moses \
    --post-process subword_nmt > raw_sys.de-en.en

cat raw_sys.de-en.en | grep -P "^D" | sort -V | cut -f 3- > sys.de-en.en
# sacrebleu -w 6 ${multi_domin_corpus}/${domain}/test.en --input sys.de-en.en