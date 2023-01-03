#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 11115 main.py --folder ./experiments/xlan
CUDA_VISIBLE_DEVICES=1 python3 main.py --folder ./experiments/xlan
