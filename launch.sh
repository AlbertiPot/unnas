export PYTHONPATH=.
time=$(date "+%Y%m%d_%H%M%S")

# search on c100 with rot
CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train_net.py \
    --cfg configs/search_based/search_phase/cifar100/rot.yaml \
    OUT_DIR result/search_rot_c100_v1rotc100seed5555_${time} \
    RNG_SEED 5555 > search_rot_c100_v1rotc100seed5555.log 2>&1 &

# search on c100 with cls
# CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/cifar100/cls.yaml \
#     OUT_DIR result/search_cls_c100_v1c100clsseed9999_${time} \
#     RNG_SEED 9999 > search_cls_c100_v1c100clsseed9999.log 2>&1 &

# eval on c100
# CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/eval_phase/cls/cifar100_custom_dartsarch.yaml \
#     OUT_DIR result/eval_c100_v2c10org_${time} \
#     RNG_SEED 0 > eval_c100_v2c10org.log 2>&1 &

################# debug
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
#     --cfg configs/search_based/search_phase/cifar100/rot.yaml \
#     OUT_DIR result/rot_debug \
#     RNG_SEED 9999 # > rot_debug.log 2>&1 &
