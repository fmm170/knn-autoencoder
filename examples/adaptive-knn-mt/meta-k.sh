domain="it"

num_neighbors=8
temperature=10

knn_models=../../knn_models
data_bin=../knnmt/multi-domain-data-bin/${domain}
datastore=../knnmt/multi-domain-datastore/${domain}
datastore_size=3449918
checkpoint=../knnmt/pretrained_model/wmt19.de-en.ffn8192.pt
save_dir=./trained_model/${domain}

mkdir -p ${save_dir}

fairseq-train ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_adaptive_knn \
    --source-lang de \
    --target-lang en \
    --arch transformer_wmt_en_de_big \
    --dropout 0.2 \
    --encoder-ffn-embed-dim 8192 \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --finetune-from-model ${checkpoint} \
    --validate-interval-updates 100 \
    --save-interval-updates 100 \
    --keep-interval-updates 1 \
    --max-update 5000 \
    --validate-after-updates 1000 \
    --save-interval 10000 \
    --validate-interval 100 \
    --keep-best-checkpoints 1 \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --no-save-optimizer-state \
    --train-subset valid \
    --valid-subset valid \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.001 \
    --batch-size 32 \
    --update-freq 1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-08 \
    --stop-min-lr 3e-05 \
    --lr 0.0003 \
    --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau \
    --lr-patience 5 \
    --lr-shrink 0.5 \
    --patience 30 \
    --max-epoch 500 \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
    --temperature-value ${temperature} \
    --save-dir ${save_dir}