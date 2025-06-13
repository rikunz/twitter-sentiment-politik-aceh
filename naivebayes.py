import re
import emoji
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import pandas as pd
import nltk
nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Pastikan stopwords bahasa Indonesia tersedia
try:
    stopwords.words('indonesian')
except OSError:
    nltk.download('stopwords')

class NaiveBayes:
    def __init__(self, model_path, vectorizer_path=None, normalisasi_path=None):
        from joblib import load
        self.model = load(model_path)
        self.labels = ['negatif', 'netral', 'positif']
        self.vectorizer = load(vectorizer_path)

        # Kamus normalisasi kata
        self.normalizad_word_dict = {}
        if normalisasi_path is not None:
            normalizad_word = pd.read_excel(normalisasi_path)
            for _, row in normalizad_word.iterrows():
                if row[0] not in self.normalizad_word_dict:
                    self.normalizad_word_dict[row[0]] = row[1]
        self.stop_words = set(stopwords.words('indonesian'))
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def repeatcharClean(self, text):
        character = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:-?!()[]{}<>"\'#/\\@')
        for c in character:
            charac_long = 5
            while charac_long > 2:
                char = c * charac_long
                text = text.replace(char, c)
                charac_long -= 1
        return text

    def clean_review(self, text):
        # Lowercase
        text = text.lower()
        text = re.sub(r'\n', ' ', text)
        # Hapus emoji
        text = emoji.demojize(text)
        text = re.sub(':[A-Za-z_-]+:', ' ', text)
        # Hapus emoticon
        text = re.sub(r"([xX;:]'?[dDpPvVoO3)(])", ' ', text)
        # Hapus link
        text = re.sub(r"(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", "", text)
        # Hapus username
        text = re.sub(r"@[\w_]+", ' ', text)
        text = re.sub(r"username", ' ', text)
        # Hapus hashtag
        text = re.sub(r'#(\S+)', '', text)
        # Hapus angka dan simbol
        text = re.sub('[^a-zA-Z,.?!]+', ' ', text)
        # Hapus karakter berulang
        text = self.repeatcharClean(text)
        text = re.sub('[ ]+', ' ', text)
        return text.strip()

    def normalize_tokens(self, tokens):
        return [self.normalizad_word_dict.get(word, word) for word in tokens]

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def preprocess(self, X):
        text = self.clean_review(X)
        tokens = word_tokenize(text)
        tokens = self.normalize_tokens(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        return ' '.join(tokens)

    def predict(self, X):
        if isinstance(X, str):
            X = [X]
        X_prep = [self.preprocess(x) for x in X]
        X_vec = self.vectorizer.transform(X_prep)
        prediksi = self.model.predict(X_vec)
        return self.labels[prediksi[0]]

