#!/bin/sh

# zh_en en_fr
CUDA_VISIBLE_DEVICES=0 python main.py --data_dir "data/dbp_wd" --ratio 0.3
CUDA_VISIBLE_DEVICES=0 python recip.py --data_dir "data/dbp_wd" --method "ralign"