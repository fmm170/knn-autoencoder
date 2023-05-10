domain="it"

num_neighbors="selected num_neighbors"
temperature="selected temperature"

knn_models=/path/to/knn_models
data_bin=/path/to/multi-domin-data-bin/${domain}
pruned_datastore=/path/to/pruned-datastore/${domain}
datastore_size="datastore size after pruning"
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt
save_dir=/path/to/trained_model/${domain}

mkdir -p ${save_dir}

CUDA_VISIBLE_DEVICES=0 fairseq-train ${data_bin} \
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
    --datastore ${pruned_datastore} \
    --datastore-size ${datastore_size} \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
    --temperature-value ${temperature} \
    --save-dir ${save_dir}