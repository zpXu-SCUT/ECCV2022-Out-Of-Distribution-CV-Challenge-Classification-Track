# # 1st stage : infer 2 models
# # CUDA_VISIBLE_DEVICES=0,1 sh test.sh

python test.py --model deit3_large_patch16_384 \
              --num-gpu 2 \
              --img-size 384 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/deit/large_384/20220911/deit3_large_0913/checkpoint-0911.pth.tar \
              --output_path ../predict-result/deit_large_384/ \
              --scoreoutput_path ../predict-result/deit_large_384/

python test.py --model convnext_large \
              --num-gpu 2 \
              --img-size 224 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/convnext/large_224/20220902/convnext_large_0904/checkpoint-0902.pth.tar \
              --output_path ../predict-result/convnext_large_224/ \
              --scoreoutput_path ../predict-result/convnext_large_224/

# 2nd stage : infer 2 models
# CUDA_VISIBLE_DEVICES=0,1 sh test.sh

python test.py --model deit3_large_patch16_384 \
              --num-gpu 2 \
              --img-size 384 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/deit/large_384/20220919/deit3_large_0928/checkpoint-0920.pth.tar \
              --output_path ../predict-result/deit_large_384_Add_5_Pseduo/ \
              --scoreoutput_path ../predict-result/deit_large_384_Add_5_Pseduo/

python test.py --model convnext_large \
              --num-gpu 2 \
              --img-size 224 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/convnext/large_224/20220925/convnext_large-224_0925/checkpoint-0925.pth.tar \
              --output_path ../predict-result/convnext_large_Add_5_Pseduo/ \
              --scoreoutput_path ../predict-result/convnext_large_Add_5_Pseduo/

# 3rd stage : infer 2 model
# CUDA_VISIBLE_DEVICES=0,1 sh test.sh
python test.py --model deit3_large_patch16_384 \
              --num-gpu 2 \
              --img-size 384 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/deit/large_384/20220930/deit3_large_0930/checkpoint-0930.pth.tar \
              --output_path ../predict-result/deit_large_384_Add_8_Pseduo/ \
              --scoreoutput_path ../predict-result/deit_large_384_Add_8_Pseduo/

python test.py --model convnext_large \
              --num-gpu 2 \
              --img-size 224 \
              --num-classes 10 \
              --batch-size 200 \
              --checkpoint ../../output/convnext/large_224/20220930/convnext_large_0930/checkpoint-0930.pth.tar \
              --output_path ../predict-result/convnext_large_Add_8_Pseduo/ \
              --scoreoutput_path ../predict-result/convnext_large_Add_8_Pseduo/



