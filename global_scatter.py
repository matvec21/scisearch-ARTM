# %%

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

MODEL = 'plsa'
LANGUAGE = 'EN'

def length(x : np.ndarray) -> float:
    ln = np.sqrt(np.power(x, 2).sum())
    return ln if ln else 1

def cos_sim(themes1 : np.ndarray, themes2 : np.ndarray) -> float:
    return np.dot(themes1, themes2) / length(themes2) / length(themes1)

# %%

plt.figure(figsize = (10, 10))

folder = 'data' + LANGUAGE.upper()
dir_path = Path(folder)

for filepath in dir_path.iterdir():
    if not filepath.name.endswith(MODEL.lower() + '_theta'):
        continue
    PATH = str(filepath)
    print('Working with', filepath)

    with open(PATH[:-10] + 'data_scores', 'r', encoding = 'utf-8') as file:
        scores = [float(i.split(' | ')[1]) for i in file.read().split('\n') if i]
    theta = np.load(PATH)

    cos_sims = [cos_sim(theta[0], topics) for topics in theta[1:]]
    scores = scores[1:]

    idx = np.random.randint(0, len(scores), size = (25, ))

    plt.scatter([scores[i] for i in idx], [cos_sims[i] for i in idx], color = 'red' if LANGUAGE == 'RU' else 'blue', alpha = 0.1, rasterized = True)

plt.title(f'Язык: {LANGUAGE.upper()}, Модель: {MODEL.upper()}', fontsize = 16)
plt.xlabel('scores')
plt.ylabel('Cos sim')
plt.grid()
plt.tight_layout()
plt.show()

# %%
