# my_domain="medical"
domain="law"
# datastore_domain="wiki"
data_bin=./multi-domain-data-bin/${domain}
datastore=./multi-domain-datastore/${domain}
datastore_size=18857646
root=/users10/lhuang/xiaokenaifan/knn-models/knn_models_cli

python $root/build_faiss_index.py \
    --use-gpu \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 64 \
    --keys-dtype fp16 \
    --knn-fp16