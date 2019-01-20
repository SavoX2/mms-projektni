import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import re

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