import argparse
import torch 
import h5py
import datasets
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader
import tracemalloc

from transformers import AutoTokenizer, PreTrainedTokenizer, XGLMForCausalLM, AutoModelForCausalLM, set_seed, DataCollatorForLanguageModeling


parser = argparse.ArgumentParser('Hidden representation saver')
parser.add_argument('--subset-size', type=int, default=200)
parser.add_argument('--batch-size', type=int, required=True)
parser.add_argument('-f', '--filename', required=True)
parser.add_argument('--model-path', default="facebook/xglm-564M")

args = parser.parse_args()

MODEL_NAME = args.model_path
DATASET_NAME = "facebook/flores"
SUBSET_SIZE:int = args.subset_size
BATCH_SIZE = args.batch_size #If BATCH SIZE is not divisible by BATCH_SLICE_SIZE, a floor operation will be applied
TEXT_ENCODING = 'utf-8'
FILENAME = args.filename
SPLIT_NAME = 'devtest'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device('cpu')

# this is the minimal set of languages that you should analyze
# feel free to experiment with additional lanuages available in the flores dataset
LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "deu_Latn",
    "arb_Arab",
    "tam_Taml",
    "quy_Latn",
    "ita_Latn"
]

# construct a pytorch data loader for each dataset
def get_tokenizer(model_name:str):
    # load a pre-trained tokenizer from the huggingface hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def get_dataloaders_for_model(datasets:dict[str, datasets.Dataset], model_name:str, batch_size:int):
    # specify the tokenization function
    tokenizer = get_tokenizer(model_name)
    def tokenization(example):
        return tokenizer(example["sentence"])
    datasets_for_model = {}
    for l in datasets:
        datasets_for_model[l] = datasets[l].map(tokenization)
        datasets_for_model[l].set_format(type='torch', columns=['input_ids', 'attention_mask'])
    hugging_collate = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return {l: DataLoader(datasets_for_model[l], batch_size, shuffle=True, collate_fn=hugging_collate) for l in LANGUAGES} #type:ignore

def print_ram_usage(program_section_name=None):
    current, peak = tracemalloc.get_traced_memory()
    if program_section_name is not None: 
        print(program_section_name)
    print(f'Current RAM usage: {current / (1024*1024)} MiB, Max: {peak / (1024*1024)} MiB')

########################################################
# Entry point
########################################################
if __name__ == "__main__":
    tracemalloc.start()
    print(f'Device: {DEVICE}')
    set_seed(42)

    sentence_datasets:dict[str, datasets.Dataset] = {l : load_dataset(DATASET_NAME, l, split=SPLIT_NAME, trust_remote_code=True, keep_in_memory=False) for l in LANGUAGES} #type:ignore

    print_ram_usage("Loaded datasets")
    dataloaders_for_xglm = get_dataloaders_for_model(sentence_datasets, MODEL_NAME, SUBSET_SIZE)
    print_ram_usage("Dataloaders created")


    model_xglm:XGLMForCausalLM = XGLMForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True) #type:ignore
    model_xglm.eval()
    model_xglm.to(device=DEVICE) #type:ignore
    xglm_tokenizer:PreTrainedTokenizer = get_tokenizer(MODEL_NAME)
    excluded_tokens = xglm_tokenizer.all_special_ids
    excluded_tokens.remove(xglm_tokenizer.bos_token_id)
    print_ram_usage("Model loaded")
    
    with torch.no_grad(): #fundamental, or you build computational graph and quickly exhaust all memory
        with h5py.File(FILENAME, mode='w') as file: #For saving to disk
            file.attrs.update(
                    model_name = MODEL_NAME,
                    dataset_name = DATASET_NAME,
                    text_encoding = TEXT_ENCODING,
                    excluded_tokens = excluded_tokens
                )
            for lang in LANGUAGES:
                print_ram_usage("Beginning of loop")
                batch = next(iter(dataloaders_for_xglm[lang])) #Just one mini-batch
                batch.to(device=DEVICE)
                print_ram_usage("Loaded batch")
                hidden_list = []
                for i in range(0, SUBSET_SIZE, BATCH_SIZE):
                    output = model_xglm.forward(**batch[i:i+BATCH_SIZE])
                    # print_ram_usage("Output calculated")
                    hidden = torch.stack(output.hidden_states, dim=0)
                    hidden_list.append(hidden.cpu())
                    
                batch = batch.to(device=DEVICE_CPU)
                hidden = np.concatenate(hidden_list, axis=1)

                print_ram_usage("Hidden representation tensor computed")
                input_ids:torch.Tensor = batch['input_ids']
                
                #Match hidden dimensions
                special_tokens_mask:torch.Tensor = torch.all(input_ids.reshape( input_ids.shape+(1,)) != torch.tensor(excluded_tokens).reshape((1,1,-1)), dim=-1)
                special_tokens_mask_broadcasted = special_tokens_mask.reshape((1,) + special_tokens_mask.shape  + (1,)).broadcast_to(hidden.shape)
                print_ram_usage("Special tokens mask broadcasted")

                #Sentence mean pooling over non-padding token of every sequence
                hidden_sentence_mean = np.mean(hidden, axis=-2, where=special_tokens_mask_broadcasted) #Averaging across token dimension ignoring all NaN values

                #Mean token representations
                unique_tokens, inverse_indices = np.unique(input_ids, return_inverse=True)
                hidden_per_token:dict[int, list[np.ndarray]] = {token: [] for token in unique_tokens}
                flattened_hidden = hidden.reshape((hidden.shape[0], -1, hidden.shape[-1]))
                for i, unique_idx in enumerate(inverse_indices):
                    hidden_per_token[unique_tokens[unique_idx]].append(flattened_hidden[:, i]) #Flatten sentence and token dimensions to the same one
                for token in unique_tokens:
                    hidden_per_token[token] = np.stack(hidden_per_token[token], axis=0).mean(axis=0)
                hidden_token_mean = np.stack(list(hidden_per_token.values()), axis=1)
                hidden_per_token = None
                print_ram_usage("Hidden token mean calculated")

                #Save everything on disk
                file.create_dataset(f'{lang}/hidden', data=hidden)
                file.create_dataset(f'{lang}/hidden_sentence_mean', data=hidden_sentence_mean)
                file.create_dataset(f'{lang}/hidden_token_mean', data=hidden_token_mean)
                file.create_dataset(f'{lang}/input_ids', data=input_ids)
                file.create_dataset(f'{lang}/unique_ids', data=unique_tokens)
                print_ram_usage("Saved all to disk")
    tracemalloc.stop()
