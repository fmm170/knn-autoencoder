domain="it"

data_bin=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/multi-domain-data-bin/${domain}
datastore=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/multi-domain-datastore/${domain}
datastore_size=18857646

reduced_keys_dimension=64
transformed_datastore=/users10/lhuang/xiaokenaifan/knn-models/examples/knnmt/transformed-datastore/${domain}
vocab_size=42024

mkdir -p ${transformed_datastore}

reduce_datastore_dims \
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