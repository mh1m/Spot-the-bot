'''

Import

'''

import re
import nltk
import numpy as np
import pandas as pd
from razdel import sentenize
from natasha import Doc, MorphVocab, Segmenter, NewsEmbedding, NewsMorphTagger, NewsNERTagger
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize



def _remove_html_tags(text: str) -> str:
    '''
    Remove HTML tags
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    text : str
    '''
    clean = re.compile("<.*?>")
    return re.sub(clean, '', text)

def _remove_http_urls(text: str) -> str:
    '''
    Remove HTTP urls
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    text : str
    '''
    return re.sub(r"http\S+", "", text)

def _remove_www_urls(text: str) -> str:
    '''
    Remove WWW urls
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    text : str
    '''
    return re.sub(r"www\S+", "", text)

def _remove_special_chars(text: str, chars: list) -> str:
    '''
    Remove special chars
    
    Parameters
    ----------
    text : str
    chars : list
    
    Returns
    -------
    text : str
    '''
    for char in chars:
        text = text.replace(char, '')
    return text.replace(u'\xa0', u' ')

def remove_web(text: str, chars: list) -> str:
    '''
    Remove web from text
    
    Parameters
    ----------
    text : str
    chars : list
    Special chars to remove
    
    Returns
    -------
    text : str
    '''
    text = _remove_html_tags(text)
    text = _remove_http_urls(text)
    text = _remove_www_urls(text)
    text = _remove_special_chars(text, chars)
    return text

def process_book(book_path: str, chars: list) -> str:
    '''
    Read book with simple processing
    
    Parameters
    ----------
    book_path : str
    chars : list
    Special chars to remove
    
    Returns
    -------
    book : str
    '''
    try:
        with open(book_path, encoding='utf-8') as f:
            book = f.read()
    except UnicodeDecodeError:
        return None
    book_text = remove_web(book, chars)
    return book_text


'''

Cleaning

'''

def _ru_clean_text_for_sentenize(text: str) -> str:
    '''
    Clean text to sentence
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    text : str
    Text with punktuation
    '''
    text = ' '.join(re.findall(r"[А-Яа-я\?|\!|\.|\…|\"|\-|\:|\–|\;]+", text))
    text = re.sub(r"(?<=[.…,?!\":–;])(?=[^\s])", r' ', text)
    return text

def _ru_text_sentenize(text: str) -> list:
    '''
    Text sentenize
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    sentences : list
    List of sentences with punktuation
    '''
    sentences = list(map(lambda sentence: sentence.text, sentenize(text)))
    return sentences
    
def _ru_split_upper(word: str) -> list:
    '''
    Split upper words
    
    Parameters
    ----------
    word or text : str
    
    Returns
    -------
    Upper words : list
    List of splitted words
    '''
    upper_word_list = re.split("(?=[А-Я])", word)
    if upper_word_list[0]:
        return upper_word_list
    return upper_word_list[1:]

def _delete_short_long_words(text: str, length_threshold: tuple) -> str:
    '''
    Delete short words
    
    Parameters
    ----------
    text : str
    length_threshold : tuple
    Min and max thresholds
    
    Returns
    -------
    text : str
    List of words with length more than length_threshold
    '''
    text = text.split()
    output = []
    for word in text:
        if len(word) > length_threshold[0] and len(word) < length_threshold[1]:
            output.append(word)
    return ' '.join(output)

def _ru_clean_sentence(sentence: str, length_threshold: tuple) -> str:
    '''
    Clean sentence
    
    Parameters
    ----------
    sentence : str
    length_threshold: int
    
    Returns
    -------
    sentence : str
    Clean sentence without punktuation, short words. All words are lower.
    '''
    sentence = ' '.join(re.findall(r"[А-Яа-я]+", sentence))
    sentence = ' '.join(_ru_split_upper(sentence))
    sentence = sentence.lower()
    sentence = _delete_short_long_words(sentence, length_threshold)
    return ' '.join(sentence.split())

def _natasha_ner_clean(text: str) -> str:
    '''
    Clean text from NER
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    text : list
    Text without NERs
    '''
    text_without_ner = text
    segmenter = Segmenter()
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    for span in doc.ner.spans:
        span_text = text[span.start:span.stop]
        text_without_ner = text_without_ner.replace(span_text, '')
    return text_without_ner

def ru_clean_text(text: str, length_threshold: tuple) -> list:
    '''
    Clean text
    
    Parameters
    ----------
    text : str
    length_threshold: int
    
    Returns
    -------
    text : list
    Clean list of sentences without punktuation, short words. All words are lower.
    '''
    text_without_ner = _natasha_ner_clean(text)
    text_for_sentenize = _ru_clean_text_for_sentenize(text_without_ner)
    sentences = _ru_text_sentenize(text_for_sentenize)
    clean_sentences = []
    for sentence in sentences:
        clean_sentence = _ru_clean_sentence(sentence, length_threshold)
        if clean_sentence:
            clean_sentences.append(clean_sentence)
    return clean_sentences

def natasha_lemmatize(text: str) -> str:
    '''
    Lemmatize text
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    text : str
    Lemmatized text
    '''
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return ' '.join([_.lemma for _ in doc.tokens])
    
def delete_stop_words(text: str, stopwords: list) -> str:
    '''
    Delete stop words
    
    Parameters
    ----------
    text : str
    stopwords : list
    
    Returns
    -------
    text : str
    Text without stopwords
    '''
    output = []
    for word in text.split():
        if word not in stopwords:
            output.append(word)
    return ' '.join(output)

def ru_preprocess_text(text: str, stopwords: list, length_threshold: int) -> list:
    '''
    Preprocess text
    
    Parameters
    ----------
    text : str
    stopwords : list
    length_threshold : int
    
    Returns
    -------
    text : list
    Preprocessed list of sentences (lemmatized, without stopwords, short words and punktuation)
    '''
    text = ' . '.join(ru_clean_text(text, length_threshold))
    text = natasha_lemmatize(text)
    output = []
    for sentence in text.split(sep=' . '):
        sentence = delete_stop_words(sentence, stopwords)
        if sentence:
            output.append(sentence)
    return output

def _eng_clean_text_for_sentenize(text: str) -> str:
    '''
    Clean text to sentence
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    text : str
    Text with punktuation
    '''
    text = ' '.join(re.findall(r"[A-Za-z\?|\!|\.|\…|\"|\-|\:|\–|\;]+", text))
    text = re.sub(r"(?<=[.…,?!\":–;])(?=[^\s])", r' ', text)
    return text

def _eng_text_sentenize(text: str) -> list:
    '''
    Text sentenize
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    sentences : list
    List of sentences with punktuation
    '''
    return sent_tokenize(text)

def _eng_split_upper(word: str) -> list:
    '''
    Split upper words
    
    Parameters
    ----------
    word or text : str
    
    Returns
    -------
    Upper words : list
    List of splitted words
    '''
    upper_word_list = re.split("(?=[A-Z])", word)
    if upper_word_list[0]:
        return upper_word_list
    return upper_word_list[1:]

def _eng_clean_sentence(sentence: str, length_threshold: int) -> str:
    '''
    Clean sentence
    
    Parameters
    ----------
    sentence : str
    length_threshold: int
    
    Returns
    -------
    sentence : str
    Clean sentence without punktuation, short words. All words are lower.
    '''
    sentence = ' '.join(re.findall(r"[A-Za-z]+", sentence))
    sentence = ' '.join(_eng_split_upper(sentence))
    sentence = sentence.lower()
    sentence = _delete_short_words(sentence, length_threshold)
    return ' '.join(sentence.split())

def eng_clean_text(text: str, length_threshold: int) -> str:
    '''
    Clean text
    
    Parameters
    ----------
    text : str
    length_threshold: int
    
    Returns
    -------
    text : list
    Clean list of sentences without punktuation, short words. All words are lower.
    '''
    text_for_sentenize = _eng_clean_text_for_sentenize(text)
    sentences = _eng_text_sentenize(text_for_sentenize)
    clean_sentences = []
    for sentence in sentences:
        clean_sentence = _eng_clean_sentence(sentence, length_threshold)
        if clean_sentence:
            clean_sentences.append(clean_sentence)
    return clean_sentences

def nltk_pos_tagger(tag: str) -> str:
    '''
    Get nltk word tag
    
    Parameters
    ----------
    tag : str
    
    Returns
    -------
    nltk_tag : list
    Returns nltk tag or None
    '''
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence: str) -> str:
    '''
    Lemmatize text
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    text : str
    Lemmatized text
    '''
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        else:
            lemmatized_sentence.append(word)
    return " ".join(lemmatized_sentence)

def eng_preprocess_text(text: str, stopwords: list, length_threshold: int) -> str:
    '''
    Preprocess text
    
    Parameters
    ----------
    text : str
    stopwords : list
    length_threshold : int
    
    Returns
    -------
    text : list
    Preprocessed list of sentences (lemmatized, without stopwords, short words and punktuation)
    '''
    text = ' . '.join(eng_clean_text(text, length_threshold))
    output = []
    for sentence in text.split(sep=' . '):
        sentence = lemmatize_sentence(sentence)
        sentence = delete_stop_words(sentence, stopwords)
        if sentence:
            output.append(sentence)
    return output

def select_text_with_threshold(dataframe: pd.DataFrame,
                               column: str,
                               threshold: int) -> pd.DataFrame:
    '''
    Select text with some threshold
    
    Parameters
    ----------
    dataframe : pd.DataFrame
    column : str
    threshold : int
    
    Returns
    -------
    corpus : pd.DataFrame
    Corpus with texts. Each text have more than threshold words.
    '''
    lambda_ = lambda text: len(text.split()) > threshold
    corpus = dataframe[dataframe[column].apply(lambda_)].reset_index(drop=True)
    return corpus

def remove_sep_from_text(text: str, separator: str) -> str:
    '''
    Remove separators from text
    
    Parameters
    ----------
    text : str
    separator : str
    
    Returns
    -------
    text : pd.DataFrame
    Text without separators
    '''
    return ' '.join(text.split(sep=separator))
