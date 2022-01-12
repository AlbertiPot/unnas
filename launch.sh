export PYTHONPATH=.
time=$(date "+%Y%m%d_%H%M%S")

### search ###
# ImageNet1K
# ImageNet1K supervised classification
# CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/imagenet/cls.yaml \
#     OUT_DIR result/search_imgnet_cls_${time} > search_imgnet_cls.log 2>&1 &

# ImageNet1K self-supervised rot
# CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/imagenet/rot.yaml \
#     OUT_DIR result/search_imgnet_rot_${time} > search_imgnet_rot.log 2>&1 &

# ImageNet1K self-supervised col
# CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/imagenet/col.yaml \
#     OUT_DIR result/search_imgnet_col_${time} > search_imgnet_col.log 2>&1 &

# ImageNet1K self-supervised jig
# CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/imagenet/jig.yaml \
#     OUT_DIR result/search_imgnet_jig_${time} > search_imgnet_jig.log 2>&1 &

### evaluation ###

# IN1K cls, arch = IN22K col
# TASK=cls
# ARCH=imagenet22k_col
# OUTDIR=result/eval_${TASK}_${ARCH}_${time}/
# CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/eval_phase/${TASK}/${ARCH}.yaml \
#     OUT_DIR ${OUTDIR} > eval_imgnet_${ARCH}.log 2>&1 &

# Cifar10
CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train_net.py \
    --cfg configs/search_based/eval_phase/cls/searchonc10.yaml \
    OUT_DIR cifar10_cls_eval > eval_c10_cls.log 2>&1 &