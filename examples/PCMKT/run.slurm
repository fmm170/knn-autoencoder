#!/bin/bash
#SBATCH -J test
#SBATCH -o log/test.log                                            
#SBATCH -p compute                                                                                     
#SBATCH -w gpu26                      
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH -t 1-00:00:00  

source ~/.bashrc

# 设置运行环境
conda activate knn
# 输入要执行的命令，例如 ./hello 或 python test.py 等
# cd ../adaptive-knn-mt
# sh meta-k.sh
sh train_compact_network.sh
# sh evaluation.sh
