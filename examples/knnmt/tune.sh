#! /usr/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0
# device=$1

domain="medical"
src_lang=de
tgt_lang=en
data_dir=./multi-domain-data-bin/${domain}
task=trans/medical.de-en
tag=3.17.finetune.lr1e4
checkpoint=./pretrained_model/wmt19.de-en.ffn8192.pt

save_dir=checkpoints/$task/$tag
if [ -d $save_dir ]; then
    rm -rf $save_dir
    # echo "tag exit,please change tag"
    # exit 0
fi
mkdir -p $save_dir
cp /users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/pretrained_model/wmt19.de-en.ffn8192.pt $save_dir/checkpoint_last.pt

cp ${BASH_SOURCE[0]} $save_dir/train.sh

cmd="fairseq-train $data_dir \
    --arch transformer_wmt_en_de_big --encoder-ffn-embed-dim 8192 \
    --restore-file $checkpoint
    --share-all-embeddings -s $src_lang -t $tgt_lang \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.2 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --max-update 200200 \
    # --reset-optimizer \
    # --reset-lr-scheduler \
    --fp16 \
    --max-epoch 25 \
    --log-interval 100 --save-dir $save_dir"

# cmd="nohup "${cmd}" > $save_dir/train1.log 2>&1 &"
eval $cmd
# tail -f $save_dir/train1.log

# nohup sh train4.sh 3 > ./logs/3.10.finetune.lr1e4.log 2>&1 &


# wmt19de2en.btsample5.ffn8192.transformer_wmt_en_de_big_bsz3584_lr0.0007_dr0.2_size_updates200000_seed21_lbsm0.1_size_sa1_upsample4//finetune1