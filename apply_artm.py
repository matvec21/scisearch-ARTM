# %%

import artm
import numpy as np

import shutil
import os

from pathlib import Path

NUM_TOPICS = 32
EPS = 1e-6

def get_theta(model : artm.ARTM) -> np.ndarray:
    theta_df = model.transform(batch_vectorizer = bv, theta_matrix_type = 'dense_theta')
    sorted_cols = sorted(theta_df.columns, key = int)
    return theta_df[sorted_cols].to_numpy().T

# %%

for folder in ('dataEN', 'dataRU'):
    dir_path = Path(folder)
    for filepath in dir_path.iterdir():
        if not filepath.name.endswith('.json'):
            continue
        PATH = str(filepath) + '_'
        print('Working with', filepath)

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
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name = 'decorrelator', tau = 0.1, gamma = 0))

        last_perplexity = float('inf')
        for i in range(20):
            model.fit_offline(bv)
            perplexity = model.score_tracker['perplexity'].last_value
            if last_perplexity - perplexity < EPS:
                break

        topic_names = model.topic_names
        theta = get_theta(model)

        with open(PATH + 'artm_theta', 'wb') as f:
            np.save(f, theta)

# %%
