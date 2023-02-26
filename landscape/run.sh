# CUDA_VISIBLE_DEVICES=5,4,3,2 mpirun -n 4 python plot_surface.py --mpi --cuda --x=-0.5:0.5:26 --y=-0.5:0.5:26 \
# --model_file cifar10/trained_nets/plot_landscape_mimdarts_cls/weights.pt \
# --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot

# for evaled nets
CUDA_VISIBLE_DEVICES=1,2,3,4,5 mpirun -n 5 python plot_surface.py --mpi --cuda --x=-1:1:51 --y=-1:1:51 \
--model_file cifar10/trained_nets/eval_mimdarts_cls/weights.pt \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot