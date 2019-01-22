import numpy as np
import pandas as pd
from pathlib import Path
import time
import joblib
import re
import sys

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

# Ime fajla sa reprezentacijom ce biti rep.bin
# Ime svakog obucenog klasifikatora ce biti klasifikator.bin
# U ovom slucaju bice 25 SVC klasifikatora obucenih, tako da ce se svaki zvati zanr_SVC.bin

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

total_start_time = time.time()

df = pd.read_csv('Klasifikatori/movies_genres_en.csv', delimiter='\t',
                     index_col=0, nrows=1000)
genres = np.array(df.drop(['plot', 'title', 'plot_lang'], axis=1).columns.values)
data_x = df[['plot']].values

if not Path(genres[0] + '_SVC.bin').exists(): # ako ne postoji jedan klasifikator, nece ni ostali postojati

    for genre in genres:
        print('Pravim reprezentaciju za: ' + genre)
        print('Pocinjem praviti reprezentaciju.')

        start_time = time.time()

        data_y = df[[genre]].values

        data_x_train, data_x_test, y_train, y_test = train_test_split(
            data_x, data_y, test_size=0.3, shuffle=True, random_state=42)
        data_x_train, data_x_val, y_train, y_val = train_test_split(
            data_x_train, y_train, test_size=0.2, shuffle=True, random_state=42)

        # if vec == None: mogao bih ustediti vremena ako bih stavio u train_test_split neki 
        # random_state pa da mi stalno isto dijeli skupove, onda ne bih morao stalno nanovo
        # praviti vektorske reprezentacije plotova, samim tim bi mozda i rezultati bili konzistentniji
        # jer je uvijek ista podjela?
        if vec == None:
            vec = TfidfVectorizer(tokenizer=tokenize, strip_accents='unicode',
                                    analyzer='word', ngram_range=(1, 3), norm='l2')
            x_train = vec.fit_transform(data_x_train.ravel())
            x_test = vec.transform(data_x_test.ravel())
            x_val = vec.transform(data_x_val.ravel())

        # necu cuvati reprezentacije za svaki zanr zasebno, nema smisla

        end_time = time.time()
        print('Reprezentacije napravljene i sacuvane za: ', end_time-start_time, 's')

        max_acc = -1
        min_hamm = 1
        time_passed = sys.maxsize
        max_f1 = -1

        acc_tuple = None
        hamm_tuple = None
        f1_tuple = None
        time_tuple = None

        print('Pocinjem podesavanje hiperparametara na validacionom skupu:')
        start_time = time.time()

        kernels = ['linear', 'rbf']
        Cs = [1, 10, 50, 100, 200, 500, 1000]
        gammas = [0.1, 0.01]

        best_kernel = ''
        best_C = -1
        best_gamma = -1

        max_acc = -1

        for C in Cs:
            for kernel in kernels:
                for gamma in gammas:
                    print('C = ', C, ', kernel = ', kernel, ', gamma = ', gamma)
                    
                    inner_start_time = time.time()

                    svc_ = SVC(C=C, kernel=kernel, gamma=gamma)
                    svc_.fit(x_train, y_train)
                    prediction = svc_.predict(x_val)

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
                            max_acc, ') dala je kombinacija kernel = ', kernel, ', C =', C, ', gamma = ', gamma)
                        max_acc = acc
                        best_kernel = kernel
                        best_C = C
                        best_gamma = gamma
                        acc_tuple = (acc, hamm, f1, kernel, C, gamma, inner_time_passed)
                    if hamm < min_hamm:
                        print('Novi najbolji hemming score (', hamm, '<',
                            min_hamm, ') dala je kombinacija kernel = ', kernel, ', C =', C, ', gamma = ', gamma)
                        min_hamm = hamm
                        hamm_tuple = (acc, hamm, f1, kernel, C, gamma, inner_time_passed)
                    if f1 > max_f1:
                        print('Novi najbolji F1 score (', f1, '>',
                            max_f1, ') dala je kombinacija kernel = ', kernel, ', C =', C, ', gamma = ', gamma)
                        max_f1 = f1
                        f1_tuple = (acc, hamm, f1, kernel, C, gamma, inner_time_passed)
                    if inner_time_passed < time_passed:
                        print('Novo najbrze vrijeme (', inner_time_passed, '<',
                            time_passed, ') dala je kombinacija kernel = ', kernel, ', C =', C, ', gamma = ', gamma)
                        time_passed = inner_time_passed
                        time_tuple = (acc, hamm, f1, kernel, C, gamma, inner_time_passed)

        print('ACC:', acc_tuple)
        print('F1:', f1_tuple)
        print('HAMM:', hamm_tuple)
        print('TIME:', time_tuple)

        cf = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma)
        prediction = cf.fit(x_train, y_train).predict(x_test)

        print('Testni:')
        print('HM:', metrics.hamming_loss(y_test, prediction))
        print('AS:', metrics.accuracy_score(y_test, prediction))
        print('F1:', metrics.f1_score(y_test, prediction, average='micro'))

        end_time = time.time()
        print('Klasifikator obucen i sacuvan za zanr ', genre, ' za: ', end_time-start_time, 's')

        joblib.dump(cf, genre + '_SVC.bin')

total_end_time = time.time()
print('Svi obuceni i sacuvani za: ', total_end_time - total_start_time, 's')