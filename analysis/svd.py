import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import itertools

def load_stopwords(path=r'./stopwords-zh.txt'):
    # default stopwords ref: https://github.com/stopwords-iso/stopwords-zh/blob/master/stopwords-zh.txt
    with open(path, encoding='utf8', mode='r') as f:
        return f.read().split('\n')

def main():
    DATA_NAME = 'toy10'
    DATA_PATH = f'../data/{DATA_NAME}.csv'
    df = pd.read_csv(DATA_PATH, sep='\t')
    doc_list = [eval(x) for x in df['text'].tolist()]  # list, shape: [#doc, len_of_each_doc]
    # Chinese Tokenizing: ref: https://zhuanlan.zhihu.com/p/345346156
    # include one character as a token (in Chinese one character could be meaningful)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=load_stopwords(), min_df=2)

    term_doc_sparse = vectorizer.fit_transform([' '.join(x) for x in doc_list])  # shape: [#doc, #term]
    terms = vectorizer.get_feature_names_out()  # len: #term

    all_set = set(itertools.chain(*text_list))
    print(len(all_set), len(vec_set))
    print(all_set - vec_set)
    print('='*50)
    print(vec_set)

if __name__ == '__main__':
    main()
