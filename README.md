Reproduce the Top-v1 cs. Acc on mobilenetv2
python -m torch.distributed.launch --nproc_per_node=8 train.py /PATH/TO/ImageNet -c ./configs/mbnv2_140.yml

Reproduce the Top-v1 cs. Acc on ResNet34
python -m torch.distributed.launch --nproc_per_node=8 train.py /PATH/TO/ImageNet -c ./configs/rn34.yml

To run the TPS analysis
python TPS_analysis/tps_sum.py

