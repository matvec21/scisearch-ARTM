# %%

import json
import os
import re
import shutil

TARGET_FOLDER = 'data'

count_ru = 0
count_en = 0

def get_lang(text : str):
    russian = len(re.findall(r'[а-яА-ЯёЁ]', text))
    english = len(re.findall(r'[a-zA-Z]', text))
    return 'RU' if russian > english else 'EN'

for name in os.listdir(TARGET_FOLDER):
    if name[-5:] != '.json':
        continue

    with open(TARGET_FOLDER + '/' + name, 'r', encoding = 'utf-8') as file:
        data = json.load(file)

    for item in data:
        item['language'] = get_lang(item['abstract'])

    lang = get_lang(data[0]['abstract'])
    os.makedirs(TARGET_FOLDER + lang, exist_ok = True)

    count = count_ru if lang == 'RU' else count_en

    with open(TARGET_FOLDER + lang + '/data%i.json' % count, 'w', encoding = 'utf-8') as file:
        json.dump(data, file, ensure_ascii = False)
    shutil.copy(TARGET_FOLDER + '/' + name[:-5] + '_vecs.npy', TARGET_FOLDER + lang + '/data%i_vecs.npy' % count)

    if lang == 'RU':
        count_ru += 1
    else:
        count_en += 1

# %%
