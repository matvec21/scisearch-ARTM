# %%

import artm
import numpy as np

import shutil
import os

from pathlib import Path
from scipy import stats
from tqdm import tqdm

LANGUAGE = 'EN'
NUM_TOPICS = 32
EPS = 1e-6
TAUS = np.linspace(0, 0.2, 12)

def get_theta(model : artm.ARTM) -> np.ndarray:
    theta_df = model.transform(batch_vectorizer = bv, theta_matrix_type = 'dense_theta')
    sorted_cols = sorted(theta_df.columns, key = int)
    return theta_df[sorted_cols].to_numpy().T

def length(x : np.ndarray) -> float:
    ln = np.sqrt(np.power(x, 2).sum())
    return ln if ln else 1

def cos_sim(themes1 : np.ndarray, themes2 : np.ndarray) -> float:
    return np.dot(themes1, themes2) / length(themes2) / length(themes1)

# %%

folder = 'data' + LANGUAGE.upper()
dir_path = Path(folder)

list_rho = []

for tau in tqdm(TAUS):
    list_rho.append(list())

    for filepath in dir_path.iterdir():
        if not filepath.name.endswith('.json'):
            continue
        PATH = str(filepath) + '_'

        if os.path.exists('batches'):
            shutil.rmtree('batches')

        bv = artm.BatchVectorizer(data_path = PATH + 'data_vw',
                                data_format = 'vowpal_wabbit',
                                batch_size = 100,
                                target_folder = 'batches',
                                class_ids = {'@default_class': 1.0, '@title_class': 3.0} )
        dictionary = artm.Dictionary()
        dictionary.gather(data_path = 'batches', vocab_file_path = PATH + 'vocab_vw')

        model = artm.ARTM(num_topics = NUM_TOPICS, num_document_passes = 10, dictionary = dictionary)
        model.scores.add(artm.PerplexityScore(name = 'perplexity', dictionary = dictionary))
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name = 'decorrelator', tau = tau, gamma = 0))

        last_perplexity = float('inf')
        for i in range(20):
            model.fit_offline(bv)
            perplexity = model.score_tracker['perplexity'].last_value
            if last_perplexity - perplexity < EPS:
                break

        topic_names = model.topic_names
        theta = get_theta(model)

        if (theta == 0).all():
            continue

        with open(PATH + 'data_scores', 'r', encoding = 'utf-8') as file:
            scores = [float(i.split(' | ')[1]) for i in file.read().split('\n') if i]

        cos_sims = [cos_sim(theta[0], topics) for topics in theta[1:]]
        scores = scores[1:]

        rho, p_val = stats.spearmanr(scores, cos_sims)
        if p_val > 0.2:
            rho = 0

        list_rho[-1].append(rho)

# %%

import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))
plt.boxplot(list_rho, tick_labels = [f'{t:.2f} ({len(list_rho[i])})' for i,t in enumerate(TAUS)])
plt.xlabel('tau (число успешных проходов)')
plt.ylabel('rho')
plt.title(f'{LANGUAGE} ARTM(gamma = 0, tau, num_topics = {NUM_TOPICS})')
plt.grid(True, alpha = 0.3)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig('ARTM32RU.svg')

# %%
