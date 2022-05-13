#!/bin/bash


python3 server.py --world_size 3 &

python3 client.py --world_size 3 --rank 1 --epoch 30 &
python3 client.py --world_size 3 --rank 2 --epoch 30 &

wait
