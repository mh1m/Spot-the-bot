"""

Clean and Groupby Texts

"""

import re
import json
import nltk
import numpy as np
import pandas as pd
from devon.devon import FSMStemmer

STOPWORDS_ADD = ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaba"]

def group_texts(dataframe: pd.DataFrame, text_column_name: str) -> pd.DataFrame:
    df_page_ind = pd.DataFrame(dataframe['article_uuid'].unique(), columns=['article_uuid'])
    df_page_ind = df_page_ind.reset_index().rename(columns={'index': 'pg_index'})
    dataframe = dataframe.merge(df_page_ind, how='left', on='article_uuid')
    dataframe = dataframe[[text_column_name, 'pg_index']]
    dataframe = dataframe.groupby(by='pg_index').agg({text_column_name: ' '.join})
    df_pg = dataframe.reset_index()
    df_pg = df_pg.dropna()
    return df_pg

def _split_upper(word: str) -> list:
    upper_word_list = re.findall("[A-Z][^A-Z]*", word)
    if ''.join(upper_word_list) != word:
        return [word]
    return upper_word_list

def _check_word_len(word: str, min_len: int = 3, max_len: int = 30) -> bool:
    if len(word) >= min_len and len(word) <= max_len:
        return word
    return None

def _delete_apostrof(word: str) -> str:
    apostrofs = ["'", "ʻ", "ʼ"]
    if word in apostrofs:
        return ""
    if word[0] in apostrofs:
        word = word[1:]
    if word[-1] in apostrofs:
        word = word[:-1]
    return word

def _check_stop_words(word: str, stop_words: list) -> str:
    if word in stop_words:
        return ""
    return word

def _clean_pipline(words_list: np.ndarray) -> pd.Series:
    words_list = pd.Series(words_list)
    words_list = words_list.apply(_delete_apostrof)
    words_list = words_list.apply(lambda word: word.lower())
    words_list = words_list.apply(lambda word: _check_stop_words(word, stop_words)).dropna()
    words_list = words_list.apply(lambda word: FSMStemmer().stem(words=word)[0])
    words_list = words_list.apply(_check_word_len).dropna().reset_index(drop=True)
    words_list = words_list.apply(lambda word: _check_stop_words(word, stop_words)).dropna()
    return words_list

def clean_text(text: str, stop_words: list) -> str:
    apostrofs = ["'", "ʻ", "ʼ"]
    splited_text = re.findall(r"[A-Za-z 'ʻʼ]+", text)
    words_list = np.array([], dtype=object)
    for sentence in splited_text:
        sentence_word_list = sentence.split()
        for word in sentence_word_list:
            splited_word_list = _split_upper(word)
            words_list = np.append(words_list, splited_word_list)
    clean_words = _clean_pipline(words_list)
    return ' '.join(clean_words.values)


# Group data by page (pg_index)

dataframe = pd.read_csv("UzWikiTexts.csv")
stop_words = json.load(open("UzStopWords.json")) + STOPWORDS_ADD
df_grouped_texts = group_texts(dataframe, 'sentence')

# Since we need to get a corpus of 10,000 documents, it is permissible that each document contains at least 150 words.

lambda_ = lambda text: len(text.split()) > 150
df_grouped_texts = df_grouped_texts[df_grouped_texts['sentence'].apply(lambda_)].reset_index(drop=True)

# Clean Texts

lambda_ = lambda text: clean_text(text, stop_words)
df_grouped_texts['clean_text'] = df_grouped_texts.loc[:, 'sentence'].apply(lambda_)
df_grouped_texts = df_grouped_texts.dropna()
df_uz_corpus = df_grouped_texts[['pg_index', 'clean_text']]

# Let's do the same again. After cleaning the texts, the number of words could decrease

lambda_ = lambda text: len(text.split()) > 130
df_uz_corpus = df_uz_corpus[df_uz_corpus['clean_text'].apply(lambda_)].reset_index(drop=True)

df_uz_corpus.to_csv("UzCleanCorpus.csv", index=False)
