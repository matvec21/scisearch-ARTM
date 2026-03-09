
# %%

from pathlib import Path
import numpy as np

# %%

def process(folder : str):
    graphs = []
    dir_path = Path(folder)

    for filepath in dir_path.iterdir():
        if filepath.is_file() and filepath.name.endswith('scores'):
            with open(filepath, 'r') as f:
                graphs.append([float(i[:-1].split(' | ')[1]) for i in f.readlines()[1:]])

    return graphs

graphsRU = process('dataRU')
graphsEN = process('dataEN')

# %%

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, figsize = (10, 10))

k = 20 / len(graphsRU)
for graph in graphsRU:
    ax1.plot(graph, color = 'red', alpha = k)
ax1.set_title('Русские статьи')

k = 20 / len(graphsEN)
for graph in graphsEN:
    ax2.plot(graph, color = 'blue', alpha = k)
ax2.set_title('Английские статьи')

for ax in (ax1, ax2):
    ax.set_xlabel('Номер')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1)
    ax.grid()

plt.tight_layout()
plt.savefig('fig1.svg')

# %%

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, figsize = (10, 10))

k = 20 / len(graphsRU)
for graph in graphsRU:
    ax1.plot(graph[:20], color = 'red', alpha = k)
ax1.set_title('Русские статьи (20 ближайших к запросу)')

k = 20 / len(graphsEN)
for graph in graphsEN:
    ax2.plot(graph[:20], color = 'blue', alpha = k)
ax2.set_title('Английские статьи (20 ближайших к запросу)')

for ax in (ax1, ax2):
    ax.set_xlabel('Номер')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1)
    ax.grid()

plt.tight_layout()
plt.savefig('fig2.svg')

# %%
