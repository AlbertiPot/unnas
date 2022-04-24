export PYTHONPATH=.
time=$(date "+%Y%m%d_%H%M%S")

# search on c100 with rot
# CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/cifar100/rot.yaml \
#     OUT_DIR result/search_rot_c100_v1rotc100seed5555_${time} \
#     RNG_SEED 5555 > search_rot_c100_v1rotc100seed5555.log 2>&1 &

# search on c100 with jig
# CUDA_VISIBLE_DEVICES=3 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/cifar100/jig.yaml \
#     OUT_DIR result/search_jig_c100_v1jigc100seed2_${time} \
#     RNG_SEED 2 > search_jig_c100_v1jigc100seed2.log 2>&1 &

# search on c100 with col
# arch_config="v1colc100seed7777"
# CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/cifar100/col.yaml \
#     OUT_DIR result/search_col_c100_${arch_config}_${time} \
#     RNG_SEED 7777 > search_col_c100_${arch_config}.log 2>&1 &

# search on c100 with cls
# CUDA_VISIBLE_DEVICES=1 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/cifar100/cls.yaml \
#     OUT_DIR result/search_cls_c100_v1c100clsseed9999_${time} \
#     RNG_SEED 9999 > search_cls_c100_v1c100clsseed9999.log 2>&1 &

# eval on c100
# arch_config="v1rotc100seed9999"
# CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/eval_phase/cls/cifar100_custom_dartsarch.yaml \
#     OUT_DIR result/eval_c100_${arch_config}_cutout_trainseed0_${time} \
#     RNG_SEED 0 > eval_c100_${arch_config}_cutout.log 2>&1 &

################# debug
# CUDA_VISIBLE_DEVICES=5 python tools/train_net.py \
#     --cfg configs/search_based/eval_phase/cls/cifar100_custom_dartsarch.yaml \
#     OUT_DIR result/cutoutdebug \
#     RNG_SEED 9999
