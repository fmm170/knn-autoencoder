domain="train_ae_medical"
# datastore_domain="wiki"
# data_bin=./multi-domain-data-bin/${domain}
datastore=./multi-domain-datastore/${domain}
datastore_size=164000
root=/users10/lhuang/xiaokenaifan/knn-models/knn_models_cli

python $root/build_faiss_index.py \
    --use-gpu \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --keys-dtype fp16 \
    --knn-fp16