import numpy as np
import pandas as pd
from pathlib import Path
import time
import joblib
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
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

# Ime fajla sa reprezentacijom ce biti rep.bin
# Ime svakog obucenog klasifikatora ce biti klasifikator.bin

rep_train_name = 'rep_train.bin'
rep_test_name = 'rep_test.bin'
rep_val_name = 'rep_val.bin'
cf_name = 'mlknn.bin'
genres_name = 'genres.bin'
vec_name = 'vec.bin'

rep_train_file = Path(rep_train_name)
rep_test_file = Path(rep_test_name)
rep_val_file = Path(rep_val_name)
cf_file = Path(cf_name)
genres_file = Path(genres_name)
vec_file = Path(vec_name)

stop_words = set(stopwords.words('english'))

vec = None
cf = None
x_train = None
y_train = None
x_test = None
y_test = None
x_val = None
y_val = None
genres = None

if not rep_train_file.exists() or not rep_test_file.exists() or not rep_val_file.exists():
    print('Ne postoji reprezentacija.')
    print('Pocinjem praviti reprezentaciju.')

    start_time = time.time()

    # , nrows=50000) # 1000 dok pokrecem na laptopu, kasnije Google colab
    df = pd.read_csv('Klasifikatori/movies_genres_en.csv', delimiter='\t',
                     index_col=0, nrows=10000)
    genres = np.array(
        df.drop(['plot', 'title', 'plot_lang'], axis=1).columns.values)

    joblib.dump(genres, genres_name)

    data_x = df[['plot']].values
    data_y = df.drop(['plot', 'title', 'plot_lang'], axis=1).values

    data_x_train, data_x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.3, shuffle=True)
    data_x_train, data_x_val, y_train, y_val = train_test_split(
        data_x_train, y_train, test_size=0.2, shuffle=True)

    if not vec_file.exists():
        vec = TfidfVectorizer(tokenizer=tokenize, strip_accents='unicode',
                              analyzer='word', ngram_range=(1, 3), norm='l2')
        x_train = vec.fit_transform(data_x_train.ravel())
        x_test = vec.transform(data_x_test.ravel())
        x_val = vec.transform(data_x_val.ravel())
    else:
        vec = joblib.load(vec_name)
        x_train = vec.fit_transform(data_x_train.ravel())
        x_test = vec.transform(data_x_test.ravel())
        x_val = vec.transform(data_x_val.ravel())

    # sacuvaj i vectorizer i tfidf reprezentacije
    joblib.dump(vec, vec_name)
    joblib.dump((x_train, y_train), rep_train_name)
    joblib.dump((x_test, y_test), rep_test_name)
    joblib.dump((x_val, y_val), rep_val_name)

    end_time = time.time()
    print('Reprezentacije napravljene i sacuvane za: ', end_time-start_time, 's')

if not cf_file.exists():

    if x_train is None:  # ako je bilo koji None, svi su None
        start_time = time.time()

        arr = joblib.load(rep_train_name)
        x_train = arr[0]
        y_train = arr[1]
        arr = joblib.load(rep_test_name)
        x_test = arr[0]
        y_test = arr[1]
        arr = joblib.load(rep_val_name)
        x_val = arr[0]
        y_val = arr[1]

        end_time = time.time()
        print('Reprezentacije ucitane za: ', end_time-start_time, 's')

    best_k = -1
    max_acc = -1
    best_s = -1
    min_hamm = 1
    time_passed = sys.maxsize
    max_f1 = -1

    acc_tuple = None
    hamm_tuple = None
    f1_tuple = None
    time_tuple = None

    print('Pocinjem podesavanje hiperparametara na validacionom skupu:')
    start_time = time.time()

    for k in np.arange(1, 6):
        for s in [0.3, 0.5, 0.7, 1.0]:
            print('k = ', k, ', s = ', s)

            inner_start_time = time.time()

            classifier = MLkNN(k=k, s=s)
            prediction = classifier.fit(x_train, y_train).predict(x_val)

            inner_end_time = time.time()
            inner_time_passed = inner_end_time - inner_start_time

            hamm = metrics.hamming_loss(y_val, prediction)
            acc = metrics.accuracy_score(y_val, prediction)
            f1 = metrics.f1_score(y_val, prediction, average='micro')
            print('HM:', hamm)
            print('AS:', acc)
            print('F1:', f1)

            if acc > max_acc:
                print('Nova najbolja tacnost (', acc, '>',
                      max_acc, ') dala je kombinacija k = ', k, ', s =', s)
                max_acc = acc
                best_k = k
                best_s = s
                acc_tuple = (acc, hamm, f1, k, s, time)
            if hamm < min_hamm:
                print('Novi najbolji hemming score (', hamm, '<',
                      min_hamm, ') dala je kombinacija k = ', k, ', s =', s)
                min_hamm = hamm
                hamm_tuple = (acc, hamm, f1, k, s, time)
            if f1 > max_f1:
                print('Novi najbolji F1 score (', f1, '>',
                      max_f1, ') dala je kombinacija k = ', k, ', s =', s)
                max_f1 = f1
                f1_tuple = (acc, hamm, f1, k, s, time)
            if inner_time_passed < time_passed:
                print('Novo najbrze vrijeme (', inner_time_passed, '<',
                      time_passed, ') dala je kombinacija k = ', k, ', s =', s)
                time_passed = inner_time_passed
                time_tuple = (acc, hamm, f1, k, s, time)

    print('ACC:', acc_tuple)
    print('F1:', f1_tuple)
    print('HAMM:', hamm_tuple)
    print('TIME:', time_tuple)

    cf = MLkNN(k=best_k, s=best_s)
    prediction = cf.fit(x_train, y_train).predict(x_test)

    print('Testni:')
    print('HM:', metrics.hamming_loss(y_test, prediction))
    print('AS:', metrics.accuracy_score(y_test, prediction))
    print('F1:', metrics.f1_score(y_test, prediction, average='micro'))

    joblib.dump(cf, cf_name)

    end_time = time.time()
    print('Klasifikator obucen i sacuvan za: ', end_time-start_time, 's')

if cf is None:
    cf = joblib.load(cf_name)
if vec is None:
    vec = joblib.load(vec_name)
if genres is None:
    genres = joblib.load(genres_name)

yes_no = 'yes'

while(yes_no != 'no'):

    film = input('Unesite opis filma: ')
    film_rep = vec.transform([film])
    predicted = cf.predict(film_rep)

    res = ''

    print(predicted[0, :].toarray()[0])

    for genre, prediction in zip(genres, predicted[0, :].toarray()[0]):
        if prediction == 1:
            res += genre + ', '

    print(res[:-2])

    yes_no = input('Da li zelite jos filmova [yes/no]: ')
