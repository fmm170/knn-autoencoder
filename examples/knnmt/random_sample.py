import argparse
import random
import tqdm
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_path', type = str, help="src data path")
    parser.add_argument('--tgt_data_path', type = str, help="tgt data path")
    parser.add_argument('--size', type = int, help="read lines")
    parser.add_argument('--datastore_size', type = int, help="datastore size")
    parser.add_argument('--out_src_path', type = str, help="output src path")
    parser.add_argument('--out_tgt_path', type = str, help="output tgt path")
    args = parser.parse_args()
    random_numbers = np.random.randint(low = 0, high=args.datastore_size, size = (1, args.size), dtype=int)
    random_numbers = list(random_numbers)
    with open(args.src_data_path, 'r', encoding='utf-8') as src, \
         open(args.tgt_data_path, 'r', encoding='utf-8') as tgt, \
         open(args.out_src_path, 'w', encoding='utf-8') as out_src, \
         open(args.out_tgt_path, 'w', encoding='utf-8') as out_tgt :
        src_data_lines = list(src.readlines())
        tgt_data_lines = list(tgt.readlines())
        for i in random_numbers[0]:
            out_src.write(src_data_lines[i])
            # out_src.write("\n")
            out_tgt.write(tgt_data_lines[i])
            # out_tgt.write("\n")
    src.close()
    tgt.close()
    out_src.close()
    out_tgt.close()


