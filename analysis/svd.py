import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.extmath import randomized_svd

def load_stopwords(path=r'./stopwords-zh.txt'):
    # default stopwords ref: https://github.com/stopwords-iso/stopwords-zh/blob/master/stopwords-zh.txt
    with open(path, encoding='utf8', mode='r') as f:
        return f.read().split('\n')

def get_term_doc_matrix(data_name):
    data_path = f'../data/{data_name}.csv'
    df = pd.read_csv(data_path, sep='\t')
    docs = [eval(x) for x in df['text'].tolist()]  # list, shape: [#doc, len_of_each_doc]
    # Chinese Tokenizing: ref: https://zhuanlan.zhihu.com/p/345346156
    # include one character as a token (in Chinese one character could be meaningful)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=load_stopwords(), min_df=2)

    doc_term_sparse = vectorizer.fit_transform([' '.join(x) for x in docs])  # shape: [#doc, #term]
    term_doc_sparse = doc_term_sparse.transpose()  # shape: [#term, #doc]
    terms = vectorizer.get_feature_names_out().tolist()  # len: #term
    return term_doc_sparse, terms, docs

def main():
    term_doc_sparse, terms, docs = get_term_doc_matrix('toy100')
    term_mat, sigma, doc_mat_T = randomized_svd(term_doc_sparse, n_components=5, random_state=2021)
    doc_mat = doc_mat_T.transpose()
    # term_mat.shape: [#term, n_components]
    # doc_mat.shape : [#doc,  n_components]
    # sigma.shape   : [#n_components]

    term_doc_approx = np.matmul(np.matmul(term_mat, np.diag(sigma)), doc_mat_T)
    frob_norm = LA.norm(term_doc_sparse - term_doc_approx, ord='fro')


if __name__ == '__main__':
    main()
