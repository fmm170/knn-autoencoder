#!/bin/bash
#SBATCH -J test
#SBATCH -o log/law_64_8_0.7_5.log                                            
#SBATCH -p compute                                                                                     
#SBATCH -w gpu24                      
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH -t 1-00:00:00  

source ~/.bashrc

# 设置运行环境
conda activate knn
# 输入要执行的命令，例如 ./hello 或 python test.py 等
# cd ../adaptive-knn-mt
# sh meta-k.sh
sh evaluation.sh
# sh evaluation.sh
