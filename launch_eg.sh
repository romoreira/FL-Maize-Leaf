#!/bin/bash


python3 server.py --world_size 5 --num_workers 4 --batch_size 32 &

python3 client.py --world_size 5 --rank 1 --epoch 50 --batch_size 32 --num_workers 4 &
python3 client.py --world_size 5 --rank 2 --epoch 50 --batch_size 32 --num_workers 4 &
python3 client.py --world_size 5 --rank 3 --epoch 50 --batch_size 32 --num_workers 4 &
python3 client.py --world_size 5 --rank 4 --epoch 50 --batch_size 32 --num_workers 4 &



wait
