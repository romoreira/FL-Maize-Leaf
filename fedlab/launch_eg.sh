#!/bin/bash


python3 server.py --world_size 3 --num_workers 4 --batch_size 32 &

python3 client.py --world_size 3 --rank 1 --epoch 50 --batch_size 32 --num_workers 4 &
python3 client.py --world_size 3 --rank 2 --epoch 50 --batch_size 32 --num_workers 4 &



wait
