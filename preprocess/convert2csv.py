import itertools
import json

import pandas as pd
import tqdm


def main():
    print('Counting #Documents...')
    with open(DATASET_PATH) as f:
        number_of_docs = sum([1 for _ in f])
    print('#Documents:', number_of_docs)

    progress = tqdm.tqdm(
        unit="docs", desc='Refactoring documents...', total=number_of_docs)

    dicts = list()
    with open(DATASET_PATH, mode="r") as f:
        for line in f:
            info = json.loads(line)
            content = ' '.join(
                list(itertools.chain(*info['content']))).split(' ')
            doc = {
                'title': info['title'],  # text
                'author': info['author'],  # text
                # for data detection format https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic-field-mapping.html
                'date': info['date'],  # date
                'column': info['column'],  # text
                'file_name': info['file_name'],  # text
                # In Elasticsearch, there is no dedicated array data type. Any field can contain zero or more values by default
                # ref: https://www.elastic.co/guide/en/elasticsearch/reference/current/array.html
                # text array: ["中央军委",  "为", "炮兵", "某团", "记", "一等功"]
                'text': [x.split('_')[0] for x in content if x],
                # text array: ["ni", "p", "n", "r", "v", "n"]
                'part_of_speech': [x.split('_')[1] for x in content if x],
            }
            dicts.append(doc)
            progress.update(1)
    print(f'{number_of_docs} documents refactored')
    df = pd.DataFrame(dicts)
    save_path = f'../data/{DATA_NAME}.csv'
    df.to_csv(save_path, sep='\t', index=False)
    print(f'csv saved to {save_path}')


if __name__ == '__main__':
    # DATA_NAME = 'toy100'
    DATA_NAME = 'rmrb_2000-2015'
    DATASET_PATH = f'/work/renyi/IR-SUN/IR-CODE/IR-BACKEND/corpus/{DATA_NAME}.jsonl'
    main()
