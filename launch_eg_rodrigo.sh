#!/bin/bash


python3 server_rodrigo.py --world_size 3 --num_workers 4 --batch_size 32 &

python3 client_rodrigo.py --world_size 3 --rank 1 --epoch 5 --batch_size 32 --num_workers 4 &
python3 client_rodrigo.py --world_size 3 --rank 2 --epoch 5 --batch_size 32 --num_workers 4 &




wait
