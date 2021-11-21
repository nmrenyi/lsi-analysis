import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import itertools

def load_stopwords(path=r'./stopwords-zh.txt'):
    with open(path, encoding='utf8', mode='r') as f:
        return f.read().split('\n')

def main():
    DATA_NAME = 'toy10'
    DATA_PATH = f'../data/{DATA_NAME}.csv'
    df = pd.read_csv(DATA_PATH, sep='\t')
    text_list = [eval(x) for x in df['text'].tolist()]  # list of word list
    # Chinese Tokenizing: ref: https://zhuanlan.zhihu.com/p/345346156
    # include one character as a token (in Chinese one character could be meaningful)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

    # TODO: how to vectorize Chinese sentences
    X = vectorizer.fit_transform([' '.join(x) for x in text_list])
    ou = vectorizer.get_feature_names_out()
    vec_set = set(ou)

    all_set = set(itertools.chain(*text_list))
    print(len(all_set), len(vec_set))
    print(all_set - vec_set)


if __name__ == '__main__':
    stopwords = load_stopwords()
    main()
