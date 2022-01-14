"""

Clean and Groupby Texts

"""

import re
import json
import nltk
import numpy as np
import pandas as pd
from devon.devon import FSMStemmer


def _clean_by_length(text: str, min_len: int = 2, max_len: int = 30) -> str:
    word_list = text.split()
    result_list = []
    for word in word_list:
        if len(word) >= min_len and len(word) <= max_len:
            result_list.append(word)
    result_text = ' '.join(result_list)
    return result_text

def clean_text(text: str, stop_words: list) -> str:
    result_text_list = []
    splited_text = re.split(r"[^A-Za-z 'ʻʼ-]+", text)
    for sentence in splited_text:
        word_list = nltk.tokenize.WhitespaceTokenizer().tokenize(sentence)
        for word in word_list:
            if len(word) == 0 or word == '-':
                continue
            if word[0] == '-':
                word = word[1:]
            try:
                if word[-1] == '-':
                    word = word[:-1]
            except IndexError:
                pass
            lower_word = word.lower()
            stem_word = FSMStemmer().stem(words=lower_word)[0]
            if stem_word not in stop_words:
                result_text_list.append(stem_word)
    result_text = ' '.join(result_text_list)
    result_text = _clean_by_length(result_text)
    return result_text

def group_texts(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_page_ind = pd.DataFrame(dataframe['article_uuid'].unique(), columns=['article_uuid'])
    df_page_ind = df_page_ind.reset_index().rename(columns={'index': 'pg_index'})
    dataframe = dataframe.merge(df_page_ind, how='left', on='article_uuid')
    dataframe = dataframe[['clean_text', 'pg_index']]
    dataframe = dataframe.groupby(by='pg_index').agg({'clean_text': ' '.join})
    df_pg = dataframe.reset_index()
    df_pg = df_pg.dropna()
    return df_pg

def create_textshape_data(dataframe: pd.DataFrame, text_shape: list, column: str) -> pd.DataFrame:
    data_text_shape = pd.DataFrame(columns=['text_shape', 'pages_amount'])
    for shape in text_shape:
        lambda_ = lambda text: len(text.split()) > shape
        pages_amount = dataframe[dataframe[column].apply(lambda_)].shape[0]
        data_text_shape = data_text_shape.append({'text_shape': shape,
                                                  'pages_amount': pages_amount}, ignore_index=True)
    return data_text_shape


# Clean Texts

dataframe = pd.read_csv("UzWikiTexts.csv")
stop_words = json.load(open("UzStopWords.json"))

lambda_ = lambda text: clean_text(text, stop_words)
dataframe['clean_text'] = dataframe.loc[:, 'sentence'].apply(lambda_)
dataframe = dataframe.dropna()

# Group data by page (pg_index)

df_grouped_texts = group_texts(dataframe)

create_textshape_data(df_grouped_texts, [100, 150, 300, 500, 1000], 'clean_text')

# Since we need to get a corpus of 10,000 documents, it is permissible that each document contains at least 150 words.

lambda_ = lambda text: len(text.split()) > 150
df_uz_corpus = df_grouped_texts[df_grouped_texts['clean_text'].apply(lambda_)].reset_index()

df_uz_corpus.to_csv("UzCleanCorpus.csv", index=False)
