# export PYTHONPATH=.
# time=$(date "+%Y%m%d_%H%M%S")

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

# search universal on both {cifar10/cifar100} with any tasks {jig/col/rot}
# time=$(date "+%Y%m%d_%H%M%S")
# PYTHONHOME=/root/miniconda3/envs/rookie/bin
# WDIR=/data/gbc/Workspace/unnas
# export PYTHONPATH=${WDIR}
# ${PYTHONHOME}/pip install -r ${WDIR}/requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
# CUDA_VISIBLE_DEVICES=0

# SEED=2
# TASK=jig
# DATASET=cifar10

# ${PYTHONHOME}/python ${WDIR}/tools/train_net.py --cfg ${WDIR}/configs/search_based/search_phase/${DATASET}/${TASK}.yaml \
#     OUT_DIR ${WDIR}/result/search_v1${TASK}${DATASET}seed${SEED}_${time} \
#     RNG_SEED ${SEED} > /log/search_v1${TASK}${DATASET}seed${SEED}_${time}.log 2>&1

# eval on c100 on mindfree
# WDIR=/data/gbc/Workspace/unnas
# PYTHONHOME=/root/miniconda3/envs/rookie/bin
# apt-get -y --no-install-recommends --allow-unauthenticated install git
# cd ${WDIR} && git checkout main
# git status
# export PYTHONPATH=${WDIR}
# ${PYTHONHOME}/pip install -r ${WDIR}/requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
# time=$(date "+%Y%m%d_%H%M%S")
# arch_config="v1cls_colc100seed2"
# CUDA_VISIBLE_DEVICES=0 ${PYTHONHOME}/python -u ${WDIR}/tools/train_net.py --cfg ${WDIR}/configs/search_based/eval_phase/cls/cifar100_custom_dartsarch.yaml OUT_DIR result/eval/eval_c100_${arch_config}_cutout_trainseed0_${time} RNG_SEED 0 NAS.GENOTYPE ${arch_config} >/log/eval_c100_${arch_config}.log 2>&1

# eval on c10 on mindfree
# WDIR=/data/gbc/Workspace/unnas
# PYTHONHOME=/root/miniconda3/envs/rookie/bin
# apt-get -y --no-install-recommends --allow-unauthenticated install git
# cd ${WDIR} && git checkout main
# git status
# export PYTHONPATH=${WDIR}
# ${PYTHONHOME}/pip install -r ${WDIR}/requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
# time=$(date "+%Y%m%d_%H%M%S")
# arch_config="v1cls_colc100seed2"
# CUDA_VISIBLE_DEVICES=0 ${PYTHONHOME}/python -u ${WDIR}/tools/train_net.py --cfg ${WDIR}/configs/search_based/eval_phase/cls/cifar10_custom_dartsarch.yaml OUT_DIR result/eval/eval_c10_${arch_config}_cutout_trainseed0_${time} RNG_SEED 0 NAS.GENOTYPE ${arch_config} >/log/eval_c10_${arch_config}.log 2>&1

# 相同的docker实例不用删除重新软连接
# rm /data/gbc/Workspace/unnas/pycls/datasets/data/cityscapes
# ln -s /data/xjh/data/cityscapes /data/gbc/Workspace/unnas/pycls/datasets/data/cityscapes
# /root/miniconda3/envs/rookie/bin/pip install -r /data/gbc/Workspace/unnas/requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
# export PYTHONPATH=/data/gbc/Workspace/unnas/
# /root/miniconda3/envs/rookie/bin/python /data/gbc/Workspace/unnas/tools/train_net.py --cfg /data/gbc/Workspace/unnas/configs/search_based/eval_phase/seg/cityscapes_seg.yaml OUT_DIR /data/gbc/Workspace/unnas/tmp



# export PYTHONPATH=.
# time=$(date "+%Y%m%d_%H%M%S")


# TASK='rot'
# SEED=9999
# CUDA_VISIBLE_DEVICES=0 nohup python -u tools/train_net.py \
#     --cfg configs/search_based/search_phase/imagenet/${TASK}.yaml \
#     OUT_DIR tmp/search_${TASK}_imagenet_v1${TASK}imgseed${SEED}_${time} \
#     RNG_SEED ${SEED} > search_${TASK}_imagenet_v1${TASK}imgseed${SEED}_.log 2>&1