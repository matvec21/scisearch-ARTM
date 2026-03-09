# %%

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy import stats

MODEL = 'artm'
LANGUAGE = 'EN'

def length(x : np.ndarray) -> float:
    ln = np.sqrt(np.power(x, 2).sum())
    return ln if ln else 1

def cos_sim(themes1 : np.ndarray, themes2 : np.ndarray) -> float:
    return np.dot(themes1, themes2) / length(themes2) / length(themes1)

# %%

folder = 'data' + LANGUAGE.upper()
dir_path = Path(folder)

list_rho = list()
list_pval = list()

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

    rho, p_val = stats.spearmanr(scores, cos_sims)
    list_rho.append(rho)
    list_pval.append(p_val)

list_rho = np.array(list_rho)
list_pval = np.array(list_pval)

# %%

idx = np.argsort(-list_rho)

fig, (ax1, ax2) = plt.subplots(2, figsize = (10, 10))
fig.suptitle(f'Язык: {LANGUAGE.upper()}, Модель: {MODEL.upper()}', fontsize = 16)

ax1.plot(list_rho[idx], color = 'purple')
ax1.set_title('rho')
ax2.set_ylim(-0.1, 0.5)

ax2.plot(list_pval[idx], color = 'black')
ax2.set_title('p-val')
ax2.set_ylim(0, 1)

ax2.plot([0, len(idx)], [0.05, 0.05], color = 'red', label = '0.05 (меньше значит уверен)')
ax2.legend()

for ax in (ax1, ax2):
    ax.set_xlabel('Номер')
    ax.grid()

plt.tight_layout()
plt.savefig(f'spearman{LANGUAGE}{MODEL}.svg')

# %%
