# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path

MODEL = 'artm'
LANGUAGE = 'EN'

# %%

# Вспомогательная функция для быстрой косинусной близости (векторизованная)
def get_cos_sims(theta):
    # theta[0] - вектор первого топика
    # theta[1:] - все остальные
    target = theta[0]
    others = theta[1:]
    
    # Считаем нормы (длины векторов)
    norm_target = np.linalg.norm(target)
    norm_others = np.linalg.norm(others, axis=1)
    
    # Косинусное расстояние через матричное умножение
    sims = np.dot(others, target) / (norm_others * norm_target + 1e-9)
    return sims

all_scores = []
all_cos_sims = []

folder = 'data' + LANGUAGE.upper()
dir_path = Path(folder)

# Собираем данные
for filepath in dir_path.iterdir():
    if not filepath.name.endswith(MODEL.lower() + '_theta'):
        continue
    
    PATH = str(filepath)
    with open(PATH[:-10] + 'data_scores', 'r', encoding='utf-8') as file:
        scores = [float(i.split(' | ')[1]) for i in file.read().split('\n') if i]
    
    theta = np.load(PATH)
    
    # Вычисляем всё сразу для файла
    sims = get_cos_sims(theta)
    
    all_scores.extend(scores[1:])
    all_cos_sims.extend(sims)

all_scores = np.array(all_scores)
all_cos_sims = np.array(all_cos_sims)

# %%

plt.figure(figsize=(12, 8))
# gridsize=100-200 — оптимально для детальности
hb = plt.hexbin(all_scores, all_cos_sims, gridsize=150, cmap='viridis', mincnt=1, bins='log')
cb = plt.colorbar(hb)
cb.set_label('log10(N points)')

plt.title(f'Hexbin: {LANGUAGE} | {MODEL}')
plt.xlabel('Scores')
plt.ylabel('Cos Sim')
plt.savefig(f'hexbin{LANGUAGE}{MODEL}.svg')

# %%
