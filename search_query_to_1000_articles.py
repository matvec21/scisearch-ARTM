# %%

import requests
import json
import os

from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector

import numpy as np

# %%

ELEMENTS_PER_QUERY = 1
TARGET_FOLDER = ...

QDRANT_HOST = ...
QDRANT_PORT = ...

QDRANT_COLLECTION = ...
QDRANT_LSEARCH_NAME = ...

with open('queries.txt', 'r', encoding = 'utf-8') as file:
    queries = list(i.replace('\n', '') for i in file.readlines() if i)

api_url = ...

articles = []

# %%

for query in queries:
    try:
        payload = {'elemsPerPage' : ELEMENTS_PER_QUERY, 'numPage' : 1, 'query' : query}
        response = requests.post(api_url, json = payload)
        response.raise_for_status()
        data = response.json()

        if not data['message'] == 'Поиск по статьям выполнен успешно':
            raise Exception('Поиск по статьям не удался (%s)' % query)

        articles += data['articles']

        if len(data['articles']) < ELEMENTS_PER_QUERY:
            raise Exception('Для запроса "%s" найти статьи не удалось' % query)
        print('"%s" OK!' % query)
    except Exception as e:
        print('ERROR', query, e)

# %%

len(articles)

# %%

with open('aricles_raw.json', 'w', encoding = 'utf-8') as f:
    json.dump(articles, f, ensure_ascii = False)

# %%

with open('aricles_raw.json', 'r', encoding = 'utf-8') as f:
    articles = json.load(f)

# %%

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_session():
    session = requests.Session()

    retry = Retry(
        total=3, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = get_session()
api_url = ...

def extract(item_data):
    id_art, score = item_data

    try:
        response = session.get(api_url + id_art, timeout=10)
        response.raise_for_status()
        
        data = response.json()

        if data.get('message') != 'Статья успешно получена':
            print(f"Skipping {id_art}: {data.get('message')}")
            return None

        article_data = data.get('article', {})
        article_data['score'] = score
        return article_data

    except Exception as e:
        print(f"Error extracting {id_art}: {e}")
        return None

os.makedirs(TARGET_FOLDER, exist_ok = True)

client = QdrantClient(host = QDRANT_HOST, port = QDRANT_PORT)
api_url = ...

for i in range(26, len(articles)):
    article = articles[i]
    article['score'] = 1.0

    try:
        rec_s = client.retrieve(
            collection_name = QDRANT_COLLECTION,
            ids = [article['articleId']],
            with_vectors = True
        )[0].vector[QDRANT_LSEARCH_NAME]

        vecs = np.empty((3001, len(rec_s)))
        vecs[0] = rec_s

        query_vector = NamedVector(name = QDRANT_LSEARCH_NAME, vector = rec_s)

        results = client.search(
            collection_name = QDRANT_COLLECTION,
            query_vector = query_vector,
            limit = 3000,
            score_threshold = 0.1,
            with_vectors = True
        )

        res = [result.id for result in results]
        scores = [result.score for result in results]
        for ii in range(1, 3001):
            vecs[ii] = results[ii - 1].vector[QDRANT_LSEARCH_NAME]

        batch = [article]
        tasks = list(zip(res, scores))

        with ThreadPoolExecutor(max_workers = 5) as executor:
            results_iter = executor.map(extract, tasks)
            valid_results = [r for r in results_iter if r is not None]
            batch += valid_results

        with open(TARGET_FOLDER + '/data%i.json' % i, 'w', encoding = 'utf-8') as file:
            json.dump(batch, file, ensure_ascii = False)

        with open(TARGET_FOLDER + '/data%i_vecs.npy' % i, 'wb') as f:
            np.save(f, vecs)

        print(f'Processed batch {i}, total articles: {len(batch)}')

    except Exception as e:
        print(f'{i} got an error -', e)

# %%
