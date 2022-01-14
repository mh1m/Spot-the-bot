"""

Read Uzbek Wiki

"""

import re
import glob
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from uuid import uuid4

nltk.download('punkt')


def split_keep_sep(string: str, sep: str) -> list:
    cleaned = []
    string = re.split('(%s)' % re.escape(sep), string)
    for _ in string:
        if _ != '' and _ != sep:
            cleaned.append(sep + _)
    return cleaned

def remove_html_tags(text: str) -> str:
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_special_chars(text: str, char_list: list) -> str:
    for char in char_list:
        text = text.replace(char, '')
    return text.replace(u'\xa0', u' ')

def process_wiki_file(wiki_file: str) -> pd.DataFrame:
    chars = ['\n']
    with open(wiki_file, encoding='utf-8') as f:
        content = f.read()
    articles = split_keep_sep(content, '<doc id=')
    dataframe = pd.DataFrame(columns=['article_uuid', 'sentence'])
    for article in articles:
        uuid = uuid4()
        article = remove_special_chars(remove_html_tags(article), chars)
        sentences = nltk.sent_tokenize(article)
        temp_dataframe = pd.DataFrame({'article_uuid': [uuid] * len(sentences),
                                       'sentence': sentences})
        dataframe = dataframe.append(temp_dataframe)
    return dataframe


wiki_files = []
for filename in glob.iglob("UzWiki/*/*"):
    wiki_files.append(filename)
    
dataframe = pd.DataFrame()
for file_name in tqdm(wiki_files):
    dataframe_file = process_wiki_file(file_name)
    dataframe = pd.concat([dataframe, dataframe_file])
dataframe['article_uuid'] = dataframe['article_uuid'].astype(str)

dataframe.to_csv("UzWikiTexts.csv", index=False)
