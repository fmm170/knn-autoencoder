domain="2w_test"
export CUDA_VISIBLE_DEVICES=0
preprocessed_multi_domin_corpus=./preprocess_data/${domain}
srcdict=./pretrained_model/dict.en.txt
data_bin=./multi-domain-data-bin/${domain}

fairseq-preprocess \
    --source-lang de \
    --target-lang en \
    --testpref ${preprocessed_multi_domin_corpus}/test.bpe \
    --srcdict ${srcdict} \
    --joined-dictionary \
    --workers 16 \
    --destdir ${data_bin}