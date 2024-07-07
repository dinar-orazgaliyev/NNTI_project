nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list

torchrun ${PROJECT_ROOT}/scripts/task2_plot_saving.py \
    -f hidden_representations_ia3.hdf5 \
    --save-dir plots_ia3