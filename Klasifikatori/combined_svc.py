import numpy as np
import pandas as pd
from pathlib import Path
import time
import joblib
import re
import sys
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sklearn.metrics as metrics

import joblib

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
minlen = 1

def normalize(text):
    # Funkcija koja vrši normalizaciju teksta
    text = text.lower()  # Pretvaranja u mala slova
    text = re.sub('<[^<>]+>', ' ', text)  # Uklanjanje HTML tagova
    text = re.sub('(http|https)://[^\s]*',
                  'httpaddr', text)  # Uklanjanje URL-ova
    text = re.sub('[^\s]+@[^\s]+', 'emailaddr', text)  # Uklanjanje mail adresa
    # Zamjena svih brojeva u tekstu riječju number
    text = re.sub('[0-9]+', 'number', text)
    text = re.sub('[$]+', 'dollar', text)  # Zamjana znaka $ rječju dollar
    return text


def tokenize(text):
    # Funkcija koja vrši tokenizaciju teksta na sastavne riječi
    sentences = nltk.sent_tokenize(text)
    stems = []
    for sentence in sentences:
        # Bez vodecih ili pratecih blank space-ova - strip()
        text = normalize(sentence.strip())
        tokens = nltk.word_tokenize(text)  # Tokenizacija teksta
        for token in tokens:
            stem = stemmer.stem(token)  # Stemizacija tokena
            if len(stem) > minlen:  # Zadržavanje tokena čija je dužina veća od minlen
                stems.append(stem)
    return stems

stop_words = set(stopwords.words('english'))

class CombinedBinarySVC:
    def __init__(self):
        self.cf_dictionary = {}

        df = pd.read_csv('Klasifikatori/movies_genres_en.csv', delimiter='\t',
                     index_col=0, nrows=1000)
        self.genres = np.array(df.drop(['plot', 'title', 'plot_lang'], axis=1).columns.values)
        data_x = df[['plot']].values

        data_y = df[[self.genres[0]]].values

        data_x_train, data_x_test, y_train, y_test = train_test_split(
            data_x, data_y, test_size=0.3, shuffle=True, random_state=42)
        data_x_train, data_x_val, y_train, y_val = train_test_split(
            data_x_train, y_train, test_size=0.2, shuffle=True, random_state=42)

        for genre in self.genres:
            self.cf_dictionary[genre] = None

        self.vec = TfidfVectorizer(tokenizer=tokenize, strip_accents='unicode',
                                    analyzer='word', ngram_range=(1, 3), norm='l2')
        x_train = self.vec.fit_transform(data_x_train.ravel())
        x_test = self.vec.transform(data_x_test.ravel())
        x_val = self.vec.transform(data_x_val.ravel())

        # inace bih radio samo 
        # self.vec = joblib.load('vec.bin')
        # ali ove sam radio sa 1000, a vec.bin je sacuvan sa 10000 pa errore baca
        # TODO: staviti na ovu jednu linij, a ne novo pravljenje stalno

        #%%
        path = './'
        import os
        for dirpath, dirs, files in os.walk(path):
            for f in files:
                fname = os.path.join(dirpath, f)    #Putanja do fajla
                #Izršiti učitavanje fajla i njegovu tokenizaciju
                # print(fname)
                for genre in self.genres:
                    if genre in fname:
                        # print('Nasao: ', genre)
                        self.cf_dictionary[genre] = joblib.load(fname)
                        break
        
    def predict(self, film):
        film_rep = self.vec.transform([film])

        res = ''

        for genre in self.genres:
            predicted = self.cf_dictionary[genre].predict(film_rep)
            print(predicted)
            if predicted[0] == 1:
                res += genre + ', '

        return res[:-2]

if __name__ == "__main__":
    cbsvc = CombinedBinarySVC()

    yes_no = 'yes'

    while yes_no != 'no':

        film = input('Unesite opis filma:')
        result = cbsvc.predict(film)
        print(result)

        yes_no = input('Jos filmova? [yes/no]')
