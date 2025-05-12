#!/bin/bash

python train.py \
D:/AI/Chess/archive/test79-apr2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack \
D:/AI/Chess/archive/test79-apr2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack \
 --gpus 1 \
 --threads 4 \
 --batch-size 4096 \
 --progress_bar_refresh_rate 20 \
 --smart-fen-skipping \
 --random-fen-skipping 10 \
 --features=HalfKP^ \
 --lambda=1.0 \
 --max_epochs=300
