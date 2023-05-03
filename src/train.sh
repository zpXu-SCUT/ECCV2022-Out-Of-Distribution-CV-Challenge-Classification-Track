## 1st stage : train 2 models
##CUDA_VISIBLE_DEVICES=0,1 nohup sh train.sh > nohup_deit_large_384_v7.out &
python -m torch.distributed.launch \
		--master_port 32620 \
		--nproc_per_node=2 train.py \
		-c configs/deit_large_384.yaml \
		--output ../../output/deit/large_384/20220911
#
## #CUDA_VISIBLE_DEVICES=0,1 nohup sh train.sh > nohup_convnext_large_224_v5.out &
#python -m torch.distributed.launch \
#		--master_port 32627 \
#		--nproc_per_node=2 train.py \
#		-c configs/convnext_large.yaml \
#        --output ../../output/convnext/large_224/20220910 \
#        --model-ema
#
## # 2nd stage : 1st round Pseudo-labeling
## #CUDA_VISIBLE_DEVICES=0,1 nohup sh train.sh > nohup_deit_large_384_v11.out &
#python -m torch.distributed.launch \
#		--master_port 32620 \
#		--nproc_per_node=2 train_round1.py \
#		-c configs/deit_large_384.yaml \
#        --output ../../output/deit/large_384/20220919
#
## #CUDA_VISIBLE_DEVICES=0,1 nohup sh train.sh > nohup_convnext_large_224_v14.out &
#python -m torch.distributed.launch \
#		--master_port 32627 \
#		--nproc_per_node=2 train_round1.py \
#		-c configs/convnext_large.yaml \
#        --output ../../output/convnext/large_224/20220919 \
#        --model-ema
#
## # 3rd stage : 2st round Pseudo-labeling
## #CUDA_VISIBLE_DEVICES=0,1 nohup sh train.sh > nohup_deit_large_384_v15.out &
#python -m torch.distributed.launch \
#		--master_port 32620 \
#		--nproc_per_node=2 train_round2.py \
#		-c configs/deit_large_384.yaml \
#        --output ../../output/deit/large_384/20220928
#
## #CUDA_VISIBLE_DEVICES=0,1 nohup sh train.sh > nohup_convnext_large_224_v18.out &
#python -m torch.distributed.launch \
#		--master_port 3262 \
#		--nproc_per_node=2 train_round2.py \
#		-c configs/convnext_large.yaml \
#        --output ../../output/convnext/large_224/20220928 \
#        --model-ema