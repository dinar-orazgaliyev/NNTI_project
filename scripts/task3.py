from typing import Iterable
from transformers import Trainer, TrainingArguments, XGLMForCausalLM, XGLMTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from transformers.models.xglm.modeling_xglm import XGLMModel, XGLMDecoderLayer, XGLMAttention
from datasets import Dataset as hgDataset, load_dataset, load_dataset_builder, get_dataset_split_names, get_dataset_config_names
import torch
from torch.nn import Parameter
import wandb
import argparse
import os
from LoRA import (
    add_lora_layers,
    model_freeze,
    model_unfreeze,
    merge_lora_layers,
)
from ia3 import modify_xglm_ia3

MODEL_NAME = "facebook/xglm-564M"
FINETUNING_TECHNIQUES = ['full', 'biases','lora', 'ia3']

LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "deu_Latn",
    "arb_Arab",
    "tam_Taml",
    "quy_Latn",
    "ita_Latn"
]

DS_DICT_WIKIPEDIA = {   
    'name': 'wikimedia/wikipedia',
    'primary_config': '20231101.qu',
    'split': 'train',
    'text_field': 'text',
    'filter_length': 256,
}

DS_DICT_OSCAR = {   
    'name': 'oscar',
    'primary_config': 'unshuffled_deduplicated_qu',
    'split': 'train',
    'text_field': 'text',
    'filter_length': 256,
}

DS_DICT_FLORES_QUY_ONLY = {   
    'name': "facebook/flores",
    'primary_config': "quy_Latn",
    'split': 'devtest',
    'text_field': 'sentence',
    'filter_length': -1,
}

DS_DICT_FLORES_ALL_IN_LANGUAGES = {
    'name': "facebook/flores",
    'primary_config': "quy_Latn",
    'configs': LANGUAGES,
    'split': 'devtest',
    'text_field': 'sentence',
    'filter_length': -1,
}

DS_MASTER_DICT = {
    'wikipedia': DS_DICT_WIKIPEDIA,
    'oscar': DS_DICT_OSCAR,
    'flores_quy_only': DS_DICT_FLORES_QUY_ONLY,
    'flores': DS_DICT_FLORES_ALL_IN_LANGUAGES,
}

DATASET_NAMES = list(DS_MASTER_DICT.keys())

def prepare_dataset(ds_dict:dict) -> hgDataset:
    """
    Ignores configs fields. Use prepare_dataset_dict_if_any() if you have any.
    If filter_length <= 0 then no length filtering is done

    Return: a Huggingface Dataset
    """
    ds = load_dataset(ds_dict['name'], ds_dict['primary_config'], split=ds_dict['split'], trust_remote_code=True)

    def tokenization(example):
        return tokenizer(example[ds_dict['text_field']], truncation=True)
    
    ds = ds.map(tokenization)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    filter_length = ds_dict['filter_length']
    if filter_length > 0:
        ds = ds.filter(lambda example: len(example['input_ids']) <= filter_length)

    return ds

def prepare_dataset_dict_if_any(ds_name:str) -> tuple[hgDataset | dict[str, hgDataset], None|str]:
    """
    Returns a tuple.
    First element: if ds_name has a list of configs, creates a dictionary of dataset per config, and returns it, else returns a single dataset
    Second element: if ds_name has a list of configs, returns the metric string name for evaluation loss of the primary config, else None
    """
    ds_dict = DS_MASTER_DICT[ds_name]
    true_primary_config = ds_dict['primary_config']
    if 'configs' in ds_dict:
        eval_datasets_dict = {}
        for c in ds_dict['configs']:
            partial_ds_dict = ds_dict.copy()
            partial_ds_dict.pop('configs')
            partial_ds_dict['primary_config'] = c
            eval_datasets_dict.update({c: prepare_dataset(partial_ds_dict)})
        return eval_datasets_dict, f'eval_{true_primary_config}_loss'
    else:
        return prepare_dataset(ds_dict), None

def get_model_parameters(model:XGLMForCausalLM, finetuning_technique:str):
    """
    Returns an iterable of model parameters
    """
    if finetuning_technique not in FINETUNING_TECHNIQUES:
        raise ValueError(f'Only values in {FINETUNING_TECHNIQUES} are allowed')
    match finetuning_technique:
        case 'full': 
            params = set(model.parameters())
        case 'biases':
            params = set(map(lambda p: p[1], filter(lambda p: 'bias' in p[0], model.model.named_parameters())))
            params.update(model.get_output_embeddings().parameters()) #Classification head!
        case 'lora':
            add_lora_layers(model, r=8, lora_alpha=16,lora_dropout=0.1 )
            model_freeze(model)
            params = set(model.parameters())
            # params.update(model.get_output_embeddings().parameters()) #Classification head! Necessary or not?
        case 'ia3':
            params = set()
            layers = model.model.layers
            decoder_layer:XGLMDecoderLayer
            for decoder_layer in layers:
                params.update(
                    decoder_layer.self_attn.k_proj[1].parameters(),
                    decoder_layer.self_attn.v_proj[1].parameters(),
                    decoder_layer.fc1[1].parameters()
                )
    return params

def freeze_unused_parameters(model:XGLMForCausalLM, used_parameters:Iterable[Parameter]):
    model.requires_grad_(False)
    for p in used_parameters:
        p.requires_grad_(True)

def modify_model(model:XGLMForCausalLM, finetuning_technique:str):
    """
    Modifies the model if the finetuning technique requires it
    """
    match(finetuning_technique):
        case 'ia3': 
            modify_xglm_ia3(model.model)
        case 'lora':
            merge_lora_layers(model)
            model_unfreeze(model)

########################################################
# Entry point
########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser('XGLM Finetuner')

    parser.add_argument('--checkpoint-dir', required=True) #Directory to save checkpoints in
    parser.add_argument('--finetuning-technique', required=True, choices=FINETUNING_TECHNIQUES) 
    parser.add_argument('--resume-from-last-checkpoint', action='store_true', default=False) #If present, resume from latest checkpoint
    parser.add_argument('--resume-from') #If present, path of a checkpoint to reload. Takes precedence over '--resume-from-last-checkpoint'
    parser.add_argument('--train-ds', required=True, choices=DATASET_NAMES) #Name in DS_MASTER_DICT dictionary
    parser.add_argument('--test-ds', default='flores', choices=DATASET_NAMES) #Name in DS_MASTER_DICT dictionary
    parser.add_argument('--epochs', type=int, default=5) #Ignored if max-steps is set
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--warmup-steps', type=int, default=20) #For linear scheduler with warmup
    parser.add_argument('--gradient-acc-steps', type=int, default=-1) #-1 to deactivate
    parser.add_argument('--train-batch-size', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=2)
    parser.add_argument('--force-cpu', action='store_true', default=False)
    parser.add_argument('--eval-steps', type=int, default=10)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--wandb-key', default='982edaff069ff93b33f8d6653d71fdaaaed41d48') #For logging and monitoring using the Weights and Biases platform: wanbd.ai. The default is Filippo Garosi's API key.
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    args = parser.parse_args()

    CHECKPOINT_DIR:str = args.checkpoint_dir
    FINETUNING_TECH:str = args.finetuning_technique
    TRAIN_DS_NAME:str = args.train_ds
    TEST_DS_NAME:str = args.test_ds
    RESUME_FROM_CHECKPOINT:bool|str = args.resume_from if (args.resume_from is not None) else (args.resume_from_last_checkpoint == True) #Works even if both are None, by setting it to False
    LEARNING_RATE:float = args.learning_rate
    EPOCHS:int = args.epochs
    MAX_STEPS:int = args.max_steps
    WARMUP_STEPS:int = args.warmup_steps
    TRAIN_BATCH_SIZE_PER_DEV:int = args.train_batch_size
    TEST_BATCH_SIZE_PER_DEV:int = args.test_batch_size
    GRADIENT_ACC_STEPS:int = args.gradient_acc_steps
    FORCE_CPU:bool = args.force_cpu
    EVAL_STRAT:str = "steps"
    EVAL_STEPS:int = args.eval_steps
    LOGGING_STEPS:int = args.logging_steps
    SAVE_STEPS:int = args.save_steps

    WANDB_KEY = args.wandb_key
    wandb.login("allow", WANDB_KEY)
    
    model:XGLMForCausalLM = XGLMForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = XGLMTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    modify_model(model, FINETUNING_TECH)
    print(model)

    train_ds, _ = prepare_dataset_dict_if_any(TRAIN_DS_NAME)
    test_ds, metric_for_best_model = prepare_dataset_dict_if_any(TEST_DS_NAME)

    params_to_optimize = get_model_parameters(model, FINETUNING_TECH)
    freeze_unused_parameters(model, params_to_optimize)
    optimizer = torch.optim.AdamW(params_to_optimize, lr=LEARNING_RATE, maximize=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, MAX_STEPS)
    training_args = TrainingArguments(CHECKPOINT_DIR, load_best_model_at_end=True, save_safetensors=False, metric_for_best_model=metric_for_best_model, do_train=True, do_eval=True, do_predict=False, prediction_loss_only=True, evaluation_strategy=EVAL_STRAT, eval_steps=EVAL_STEPS, logging_steps=LOGGING_STEPS, save_steps=SAVE_STEPS, per_device_train_batch_size=TRAIN_BATCH_SIZE_PER_DEV, per_device_eval_batch_size=TEST_BATCH_SIZE_PER_DEV, gradient_accumulation_steps=GRADIENT_ACC_STEPS, num_train_epochs=EPOCHS, max_steps=MAX_STEPS, no_cuda=FORCE_CPU)
    trainer = Trainer(model, training_args, collator, train_dataset=train_ds, eval_dataset=test_ds, tokenizer=tokenizer, compute_metrics=None, optimizers=(optimizer, scheduler), preprocess_logits_for_metrics=None)

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    trainer.save_model(os.path.join(CHECKPOINT_DIR, 'best_model')) #Since we use load_best_model_at_end, we're saving the best model on the eval loss