# scisearch-ARTM

apply_plsa.py - пройти по всем выборкам моделью PLSA и сохранить матрицу THETA в файлы data(num).json_plsa_theta

apply_artm.py - пройти по всем выборкам моделью ARTM и сохранить матрицу THETA в файлы data(num).json_artm_theta

find_decor_artm.py - перебрать различные tau для ARTM и построить boxplot, проход по всем выборкам несколько раз

generate_tokens_for_artm.py - обработать сырой текст из data(num).json в VW формат (data0.json_data_vw, data0.json_vocab_vw, data0.json_data_scores, data0.json_combinations.pkl)

global_kde.py - пройтись по всем выборкам и построить hexbin плотность в координатах (NN score, THETA cos sim)

global_scatter.py - то же самое, что global_kde, но scatter

global_spearman.py - для каждой выборки посчитать йоэффициент корреляции Спирмена между (NN score, THEHTA cos sim), отсортировать и отобразить в виде графика (ДЛЯ ОДНОЙ МОДЕЛИ)

global_spearman_comparison.py - global_spearman, где на одном графике линии моделей PLSA и ARTM

only_score_graph.py - график NN score

queries.txt - названия статей для поиска во время сборки датасета

data(num)_vecs.npy - scisearch NN эмбеддинги
