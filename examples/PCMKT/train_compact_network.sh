CUDA_VISIBLE_DEVICES=0
domain="medical"

data_bin=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/multi-domain-data-bin/${domain}
datastore=./multi-domain-datastore/${domain}
datastore_size=6501418

reduced_keys_dimension=64
transformed_datastore=./transformed-datastore/${domain}
vocab_size=42024

mkdir -p ${transformed_datastore}

python /users10/lhuang/xiaokenaifan/knn-models/knn_models_cli/reduce_datastore_dims.py \
    --method PCKMT \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --keys-dtype fp16 \
    --transformed-datastore ${transformed_datastore} \
    --reduced-keys-dimension ${reduced_keys_dimension} \
    --stage train_pckmt \
    --vocab-size ${vocab_size} \
    --max-epoch 1000000000 \
    --max-update 70000 \
    --keep-best-checkpoints 10 \
    --betas "(0.9, 0.98)" \
    --log-interval 10 \
    --clip-norm 1.0