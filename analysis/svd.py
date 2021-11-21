import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


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

    term_doc_sparse = vectorizer.fit_transform([' '.join(x) for x in docs])  # shape: [#doc, #term]
    terms = vectorizer.get_feature_names_out().tolist()  # len: #term
    return term_doc_sparse, terms, docs

def main():
    term_doc_sparse, terms, docs = get_term_doc_matrix('toy100')

if __name__ == '__main__':
    main()
