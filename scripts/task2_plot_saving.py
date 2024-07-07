# # Task 2: Visualize hidden represenations of a model

import itertools
import h5py
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, PreTrainedTokenizer
import argparse
import os

parser = argparse.ArgumentParser('Hidden representation plotter')
parser.add_argument('-f', '--filename', required=True)
parser.add_argument('--save-dir', default='plots')
parser.add_argument('--average-token-duplicates', default=False, action='store_true')
parser.add_argument('--tsne-perplexity', type=int, default=30)

args = parser.parse_args()

FILENAME = args.filename
SAVE_DIR = args.save_dir
N_COMPONENTS = 2
PERPLEXITY = args.tsne_perplexity
AVERAGE_TOKEN_DUPLICATES:bool = args.average_token_duplicates

# this is the minimal set of languages that you should analyze
# feel free to experiment with additional lanuages available in the flores dataset
LANGUAGES = [
    "ita_Latn",
    "eng_Latn",
    "spa_Latn",
    "deu_Latn",
    "arb_Arab",
    "tam_Taml",
    "quy_Latn",
]

COLUMN_DICT  = {
    'Z1': float, 
    'Z2': float,
    'language': str,
    'layer': int,
    'type': str,
    'id': int, 
    'str_form': str
} #For hidden space visualization
COLUMNS = list(COLUMN_DICT.keys())
def prepare_df_for_visualization(lang_column:list[str], curr_layer:int, features:np.ndarray, id_type:str, input_ids:np.ndarray, string_forms:np.ndarray|list, dtype_dict:dict) -> pd.DataFrame:
    n_datapoints = features.shape[0]
    data = {
        'Z1': features[:, 0], 
        'Z2': features[:, 1],
        'language': lang_column,
        'layer': np.full(n_datapoints, curr_layer),
        'type': np.full(n_datapoints, id_type),
        'id': input_ids,
        'str_form': string_forms
    }
    
    df = pd.DataFrame(data=data)
    return df.astype(dtype_dict)

alt.data_transformers.disable_max_rows() #To plot large datasets

with h5py.File(FILENAME, mode='r') as file:
    TEXT_ENCODING = file.attrs['text_encoding']
    MODEL_NAME = file.attrs['model_name']
    xglm_tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    excluded_tokens = np.array(file.attrs['excluded_tokens'])
    map_token_id_to_str = np.vectorize(xglm_tokenizer.decode)
    pca = PCA(n_components=N_COMPONENTS)
    tsne = TSNE(n_components=N_COMPONENTS, perplexity=PERPLEXITY)
    

    lang = LANGUAGES[0]
    hidden = file[f'{lang}/hidden']
    n_hidden_layers, n_hidden_features = hidden.shape[0], hidden.shape[-1]

    for l in range(n_hidden_layers):
        df_pca = pd.DataFrame()
        df_tsne = pd.DataFrame()
        layer_tokens_no_excluded_all = np.empty((0, n_hidden_features))
        layer_sentence_mean_all = np.empty((0, n_hidden_features))
        token_ids_no_excluded_all = np.empty(0, dtype='int64')
        sentences = []
        examples_per_lang:dict[str, tuple[int, int]] = {} #For every language, tuple of number of tokens and sentences
        for lang in LANGUAGES:
            input_ids = np.array(file[f'{lang}/input_ids'])
            unique_ids = np.array(file[f'{lang}/unique_ids'])
            hidden = file[f'{lang}/hidden']
            hidden_token_mean = file[f'{lang}/hidden_token_mean']
            hidden_sentence_mean = file[f'{lang}/hidden_sentence_mean']
                    
            n_hidden_layers, n_sentences, n_max_tokens, n_hidden_features = hidden.shape

            
            sentences.extend([xglm_tokenizer.decode(s, skip_special_tokens=True) for s in input_ids])

            if AVERAGE_TOKEN_DUPLICATES:
                unique_excluded_tokens_mask:np.ndarray = np.all(unique_ids.reshape( unique_ids.shape+(1,)) != excluded_tokens, axis=-1)
                layer_tokens = hidden_token_mean[l]
                layer_tokens_no_excluded = layer_tokens[unique_excluded_tokens_mask]

                token_ids_no_excluded = unique_ids[unique_excluded_tokens_mask]
                
            else:
                excluded_tokens_mask = np.all(input_ids.reshape( input_ids.shape+(1,)) != excluded_tokens.reshape((1,1,-1)), axis=-1)
                layer_tokens = hidden[l]
                layer_tokens_no_excluded = layer_tokens[excluded_tokens_mask]

                token_ids_no_excluded = input_ids[excluded_tokens_mask]
            
            layer_sentence_mean = hidden_sentence_mean[l] #"Excluded" tokens were already left out
            examples_per_lang[lang] = len(token_ids_no_excluded), n_sentences #Useful for plotting

            token_ids_no_excluded_all = np.concatenate([token_ids_no_excluded_all, token_ids_no_excluded], axis=0)
            layer_tokens_no_excluded_all = np.concatenate([layer_tokens_no_excluded_all, layer_tokens_no_excluded], axis=0)
            layer_sentence_mean_all = np.concatenate([layer_sentence_mean_all, layer_sentence_mean], axis=0)
        
        n_tokens_no_excluded = len(token_ids_no_excluded_all)
        layer_all = np.concatenate([layer_tokens_no_excluded_all, layer_sentence_mean_all], axis=0)
        
        #layer_all now contains all languages: use dimensionality reduction techniques and plot the layer's hidden representational space
        decoded_tokens_all = map_token_id_to_str(token_ids_no_excluded_all)
        pca_result = pca.fit_transform(layer_all)
        tsne_result = tsne.fit_transform(layer_all)
        hidden_tokens_pcad, hidden_sentences_pcad = pca_result[:n_tokens_no_excluded], pca_result[n_tokens_no_excluded:]
        hidden_tokens_tsned, hidden_sentences_tsned = tsne_result[:n_tokens_no_excluded], tsne_result[n_tokens_no_excluded:]

        lang_column_tokens, lang_column_sentences = [], []
        for lang, counts in examples_per_lang.items():
            token_count, sentence_count = counts
            lang_column_tokens.extend(itertools.repeat(lang, token_count))
            lang_column_sentences.extend(itertools.repeat(lang, sentence_count))
        
        df_layer_tokens_pca = prepare_df_for_visualization(lang_column_tokens, l, hidden_tokens_pcad, 'token', token_ids_no_excluded_all, decoded_tokens_all, COLUMN_DICT)
        df_layer_tokens_tsne = prepare_df_for_visualization(lang_column_tokens, l, hidden_tokens_tsned, 'token', token_ids_no_excluded_all, decoded_tokens_all, COLUMN_DICT)
        df_layer_sentences_pca = prepare_df_for_visualization(lang_column_sentences, l, hidden_sentences_pcad, 'sentence', -1, sentences, COLUMN_DICT)
        df_layer_sentences_tsne = prepare_df_for_visualization(lang_column_sentences, l, hidden_sentences_tsned, 'sentence', -1, sentences, COLUMN_DICT)
        
        df_pca = pd.concat([df_pca, df_layer_tokens_pca, df_layer_sentences_pca], axis=0)
        df_tsne = pd.concat([df_tsne, df_layer_tokens_tsne, df_layer_sentences_tsne], axis=0)
      
        
        selection_layer = alt.selection_single(fields=['language'], bind='legend')
        selection_type = alt.selection_single(fields=['type'], bind='legend')

        if not os.path.isdir(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        
        for df, reduction_tech in [(df_pca, 'PCA'), (df_tsne, 'T-SNE')]:
            df:pd.DataFrame
            df.astype(COLUMN_DICT)
            title = f'Layer {l}, {reduction_tech} projection'
            alt.Chart(df).mark_point().encode(
                x=alt.X('Z1', axis=alt.Axis(labelOverlap = True), type='quantitative'),
                y=alt.Y('Z2', axis=alt.Axis(labelOverlap = True), type='quantitative'),
                color=alt.Color('language', type='nominal' ),
                shape=alt.Shape('type', type='nominal'),
                tooltip=COLUMNS,
            ).properties(
                width=800,
                height=300,
                title=title
            ).add_selection(
                selection_layer
            ).add_selection(
                selection_type
            ).transform_filter(
                selection_layer
            ).transform_filter(
                selection_type
            ).interactive().save(os.path.join(SAVE_DIR, f'{title}.html'))