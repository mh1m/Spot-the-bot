"""

SVD

"""

import numpy as np
from scipy.sparse.linalg import svds

K = 1024


matrix_word_doc = np.load("MATRIX_WORD_DOC.npy")

u, sigma, vt = svds(matrix_word_doc.T, K)

np.save("U", u)
np.save("Sigma", sigma)
np.save("VT", vt)
