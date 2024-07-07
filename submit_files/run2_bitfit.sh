nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list

torchrun ${PROJECT_ROOT}/scripts/task2.py \
    --subset-size 200 \
    --batch-size 4 \
    -f hidden_representations_bitfit.hdf5 \
    --model-path ${PROJECT_ROOT}/quy_biases_finetuning.model/best_model