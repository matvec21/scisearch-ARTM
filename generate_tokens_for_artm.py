
# %%

import json
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

import pymorphy3
import nltk

def get_lang(text : str):
    russian = len(re.findall(r'[а-яА-ЯёЁ]', text))
    english = len(re.findall(r'[a-zA-Z]', text))
    return 'RU' if russian > english else 'EN'

class TextProcessor:
    def __init__(self):
        self.stem2word_counter: Dict[str, Counter] = defaultdict(Counter)

        self.stemmer = nltk.stem.PorterStemmer()
        try:
            en_stopwords = nltk.corpus.stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
            en_stopwords = nltk.corpus.stopwords.words('english')

        self.stopwords_EN = set(self.stemmer.stem(w) for w in en_stopwords)

        self.morph = pymorphy3.MorphAnalyzer()
        self.stopwords_RU = set(self.lemmatize(i.replace('\n', '')) for i in nltk.corpus.stopwords.words('russian'))

        self.regex_filter = re.compile(r'[^а-яa-z0-9\s]')

    @lru_cache(maxsize = 100000)
    def lemmatize(self, word: str) -> str:
        return self.morph.parse(word)[0].normal_form

    @lru_cache(maxsize = 100000)
    def stem(self, word: str) -> str:
        stemmed = self.stemmer.stem(word)
        self.stem2word_counter[stemmed][word] += 1
        return stemmed

    def preprocess_text_RU(self, text: str) -> List[str]:
        text = text.lower().replace('\n', ' ').replace('ё', 'е')
        text = self.regex_filter.sub('', text)

        result = []
        for word in text.split():
            if 2 < len(word) < 20:
                lemma = self.lemmatize(word)
                if lemma not in self.stopwords_RU:
                    result.append(lemma)
        return result

    def preprocess_text_EN(self, text: str) -> List[str]:
        text = text.lower().replace('\n', ' ').replace('ё', 'е')
        text = self.regex_filter.sub('', text)
        
        result = []
        for word in text.split():
            if 2 < len(word) < 20:
                stemmed = self.stem(word)
                if stemmed not in self.stopwords_EN:
                    result.append(stemmed)
        return result

    def generate_stem2word(self):
        stem2word = {
            stem: counter.most_common(1)[0][0] 
            for stem, counter in self.stem2word_counter.items()
        }
        with open('stem2word.pkl', 'wb') as file:
            pickle.dump(stem2word, file)

class DataGenerator:
    def __init__(self, data_path: Path, processor: TextProcessor, window: int = 3, max_docs: int = 1000):
        self.data_path = data_path
        self.processor = processor
        self.window = window
        self.max_docs = max_docs
        
        with open(data_path, 'r', encoding='utf-8') as file:
            self.data: List[Dict[str, Any]] = json.load(file)
            if len(self.data) > 1:
                self.data.pop(1) # БЛИЖАЙШИЙ В ПОИСКЕ ЭТО САМ ДОКУМЕНТ-ЗАПРОС

        self.combinations = defaultdict(lambda: defaultdict(int))

    def generate(self) -> int:
        docs = 0
        counter = Counter()
        
        if not self.data:
            return 0
            
        lang = get_lang(self.data[0]['abstract'])

        out_vw = f'{self.data_path}_data_vw'
        out_scores = f'{self.data_path}_data_scores'

        with open(out_vw, 'w', encoding='utf-8') as file_vw, \
             open(out_scores, 'w', encoding='utf-8') as file_score:

            for article in self.data:
                if get_lang(article['abstract']) != lang:
                    continue

                raw_title, raw_abstract = article['title'], article['abstract']
                
                if lang == 'RU':
                    abstract = self.processor.preprocess_text_RU(raw_abstract)
                    title = self.processor.preprocess_text_RU(raw_title)
                else:
                    abstract = self.processor.preprocess_text_EN(raw_abstract)
                    title = self.processor.preprocess_text_EN(raw_title)

                if not (30 <= len(abstract) <= 250): # 99% here
                    continue

                counter.update(abstract)
                counter.update(title)

                self.update_combinations(abstract)
                self.update_combinations(title)

                abstract_str = ' '.join(abstract)
                title_str = ' '.join(title)
                score = article['score']

                file_vw.write(f'doc_{docs} |@default_class {abstract_str} |@title_class {title_str}\n')
                file_score.write(f'doc_{docs} | {score:f}\n')

                docs += 1
                if docs >= self.max_docs:
                    break

        with open(f'{self.data_path}_vocab_vw', 'w', encoding='utf-8') as file_vocab:
            vocab_lines = [f'{w}\n' for w, count in counter.items() if count >= 10]
            file_vocab.writelines(vocab_lines)

        with open(f'{self.data_path}_combinations.pkl', 'wb') as file_comb:
            pickle.dump({k: dict(v) for k, v in self.combinations.items()}, file_comb)

        return docs

    def update_combinations(self, words: List[str]):
        total = len(words)
        for pos1, word1 in enumerate(words):
            self.combinations[word1]['APPEARANCES'] += 1

            start_idx = max(0, pos1 - self.window)
            end_idx = min(total, pos1 + self.window + 1)
            
            for pos2 in range(start_idx, end_idx):
                if pos1 != pos2:
                    word2 = words[pos2]
                    self.combinations[word1][word2] += 1


if __name__ == '__main__':
    processor = TextProcessor()
    count1000 = 0

    for folder in ['dataRU', 'dataEN']:
        dir_path = Path(folder)

        for filepath in dir_path.iterdir():
            if filepath.is_file() and filepath.name.endswith('.json'):
                print(f'Working with {filepath.name} in {folder}')
                generator = DataGenerator(filepath, processor)
                docs_processed = generator.generate()
                print(f'Got {docs_processed} articles! Less then thousand: {count1000}')
                if docs_processed < 1000:
                    count1000 += 1
                    print('LESS THEN 1000')

    processor.generate_stem2word()
    print('Done!')

# %%
