nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list

torchrun --nnodes 1 --nproc_per_node gpu ${PROJECT_ROOT}/scripts/task3.py \
    --train-ds wikipedia \
    --test-ds flores \
    --checkpoint-dir quy_lora_finetuning.model \
    --finetuning-technique lora \
    --eval-steps 20 \
    --logging-steps 20 \
    --max-steps 5000 \
    --learning-rate "5e-4" \
    --wandb-key 39da3e0b522996c0e377fa0f6672af6af307cc6d
