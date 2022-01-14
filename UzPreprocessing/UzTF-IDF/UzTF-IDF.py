"""

TF-IDF

"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_PATTERN = r"\S+"


def create_word_doc_matrix(corpus: pd.DataFrame, token_pattern: str) -> tuple:
    vectorizer = TfidfVectorizer(token_pattern=token_pattern)
    matrix_word_doc = vectorizer.fit_transform(corpus['clean_text'].values)
    return matrix_word_doc.toarray(), np.array(vectorizer.get_feature_names())


corpus = pd.read_csv("UzCleanCorpus.csv")
matrix_word_doc, word_list = create_word_doc_matrix(corpus, TOKEN_PATTERN)

np.save("MATRIX_WORD_DOC", matrix_word_doc)
np.save("WORD_LIST", word_list)
