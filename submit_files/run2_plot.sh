nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list

torchrun ${PROJECT_ROOT}/scripts/task2_plot_saving.py \
    -f hidden_representations.hdf5 \
    --save-dir plots_vanilla