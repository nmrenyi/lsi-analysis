import os
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.extmath import randomized_svd
import argparse
from sys import stderr
from scipy import sparse
import json

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
    term_doc_sparse, terms, docs = get_term_doc_matrix(args.dataset)
    term_mat, sigma, doc_mat_T = randomized_svd(term_doc_sparse, n_components=args.dim, random_state=args.random_seed)
    doc_mat = doc_mat_T.transpose()
    # term_mat.shape: [#term, n_components]
    # doc_mat.shape : [#doc,  n_components]
    # sigma.shape   : [#n_components]

    term_doc_approx = np.matmul(np.matmul(term_mat, np.diag(sigma)), doc_mat_T)
    frob_norm = LA.norm(term_doc_sparse - term_doc_approx, ord='fro')
    print('frobenius norm between raw and approximated:', frob_norm)

    config = f'data_{args.dataset}-dim_{args.dim}-rand_{args.random_seed}'
    save_dir = f'../result/{config}'
    os.makedirs(save_dir, exist_ok=True)

    sparse.save_npz(os.path.join(save_dir, f'termDocSparse'), term_doc_sparse)
    np.save(os.path.join(save_dir, f'termArray'), term_mat)
    np.save(os.path.join(save_dir, f'docArray'), doc_mat)
    with open(os.path.join(save_dir, 'term-doc.json'), mode='w', encoding='utf8') as f:
        json.dump({'terms': terms, 'docs': docs}, f, ensure_ascii=False)
    with open(os.path.join(save_dir, 'frobenius-norm-approx-raw.txt'), mode='w', encoding='utf8') as f:
        f.write(str(frob_norm))
    print(f'all files saved to {save_dir}')


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument('--dim', type=int, default=100, help='truncated dimension for svd, default 100')
    parser.add_argument('--random_seed', type=int, default=9999744, help='random seed, default 9999744')
    parser.add_argument('--dataset', type=str, default='toy100', help='dataset name, default toy100')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser = parse_args(parser)
    args = parser.parse_args()
    print(args, file=stderr)
    main()
