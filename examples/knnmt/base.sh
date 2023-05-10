domain="subtitles"

multi_domin_corpus=./multi_domain_new_split
DATA_PATH=./multi-domain-data-bin/${domain}
BASE_MODEL=./pretrained_model/wmt19.de-en.ffn8192.pt

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_PATH \
    --task translation \
    --path $BASE_MODEL \
    --dataset-impl mmap \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --gen-subset test \
    --model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
    --max-tokens 4096 \
    --scoring sacrebleu \
    --tokenizer moses --remove-bpe > base_raw_sys.de-en.en

cat base_raw_sys.de-en.en | grep -P "^D" | sort -V | cut -f 3- > base_sys.de-en.en
sacrebleu -w 6 ${multi_domin_corpus}/${domain}/test.en --input base_sys.de-en.en
