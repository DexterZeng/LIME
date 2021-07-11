#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main.py --data_dir "data/zh_en" --ratio 0.3
CUDA_VISIBLE_DEVICES=0 python recip.py --data_dir "data/zh_en" --method "ralign-wr"