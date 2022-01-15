"""

Vectorize Corpus

"""


import json
import numpy as np
import pandas as pd
from tqdm import tqdm

VECTOR_SHAPE = 8
SEQUENCE_SHAPE = 1


def _cut_dict_values(input_dict: dict, k: int) -> dict:
    lambda_ = lambda value: value[:k]
    dict_ = dict(zip(input_dict, map(lambda_, input_dict.values())))
    return dict_

def _concat_columns(dataframe: pd.DataFrame) -> np.ndarray:
    if dataframe.shape[0] == 0:
        return np.array([], dtype=object)
    array = dataframe.iloc[0].values
    result = [array]
    for i in range(1, dataframe.shape[0]):
        curr_array = dataframe.iloc[i].values
        result = np.append(result, [curr_array], axis=0)
    return result

def create_ngrams(sentence: list, sequence_shape: int) -> pd.Series:
    if sequence_shape > len(sentence):
        return pd.Series(dtype='object')
    n_words = len(sentence) - sequence_shape + 1
    n_sentence = sentence[:n_words]
    ngrams = pd.Series(dtype='object')
    for i in range(len(n_sentence)):
        ngrams = ngrams.append(pd.Series([sentence[i:i + sequence_shape]]))
    ngrams.reset_index(drop=True, inplace=True)
    return ngrams

def vectorize_text(text: str,
                   word_vector_dict: dict,
                   vector_shape: int,
                   sequence_shape: int) -> np.ndarray:
    word_vector_dict = _cut_dict_values(word_vector_dict, vector_shape)
    sentence = text.split()
    ngrams = create_ngrams(sentence, sequence_shape)
    dataframe_ngrams = ngrams.apply(lambda ngram: pd.Series(ngram))
    dataframe_vectors = dataframe_ngrams.apply(lambda ngram: ngram.map(word_vector_dict))
    result_dataframe = _concat_columns(dataframe_vectors)
    return result_dataframe

def vectorize_corpus(corpus_array: np.ndarray,
                     word_vector_dict: dict,
                     vector_shape: int,
                     sequence_shape: int) -> np.ndarray:
    result = list()
    for text_index in tqdm(range(len(corpus_array))):
        vectorized_text = vectorize_text(corpus_array[text_index], word_vector_dict,
                                         vector_shape, sequence_shape)
        result.append(vectorized_text)
    return np.array(result, dtype=object)


corpus = pd.read_csv("UzCleanCorpus.csv")
word_list = np.load("WORD_LIST.npy")
word_vectors = np.load("U.npy")

word_vector_dict = dict(zip(word_list, word_vectors))
vectorized_corpus = vectorize_corpus(corpus['clean_text'].values,
                                     word_vector_dict,
                                     VECTOR_SHAPE,
                                     SEQUENCE_SHAPE)

np.save("UzVectorizedCorpus", vectorized_corpus)

lambda_ = lambda vector: vector.tolist()
dict_ = dict(zip(word_vector_dict, map(lambda_, word_vector_dict.values())))
with open('WordVectorDict.json', 'w') as f:
    json.dump(dict_, f)
