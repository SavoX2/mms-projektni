# %%

import numpy as np
# from skmultilearn.dataset import load_dataset
from sklearn.model_selection import GridSearchCV
from skmultilearn.adapt import MLkNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# , nrows=50000) # 1000 dok pokrecem na laptopu, kasnije Google colab
df = pd.read_csv('Klasifikatori/movies_genres_en.csv',
                 delimiter='\t', index_col=0, nrows=50000)
genres = np.array(
    df.drop(['plot', 'title', 'plot_lang'], axis=1).columns.values)

# joblib.dump(genres, genres_name)

data_x = df[['plot']].values
data_y = df.drop(['plot', 'title', 'plot_lang'], axis=1).values

data_x_train, data_x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, shuffle=True)
data_x_train, data_x_val, y_train, y_val = train_test_split(
    data_x_train, y_train, test_size=0.2, shuffle=True)

# print('X', X_train[0].shape, type(X_train[0]))
# print('Y', y_train[0])
# print('fn', feature_names)
# print('ln', label_names)

print(y_train.shape, y_test.shape)

vec = TfidfVectorizer(strip_accents='unicode',
                      analyzer='word', ngram_range=(1, 3), norm='l2')
x_train = vec.fit_transform(data_x_train.ravel())
x_test = vec.transform(data_x_test.ravel())
x_val = vec.transform(data_x_val.ravel())

# print(np.unique(y_train.rows).shape, np.unique(y_test.rows).shape)
print(x_train.shape[1])

parameters = {'k': range(1, 6), 's': [0.5, 0.7, 1.0]}
score = 'f1_micro'

clf = GridSearchCV(MLkNN(), parameters, scoring=score)
clf.fit(x_train, y_train)

print(clf.best_params_, clf.best_score_)

# %%

yes_no = 'yes'

# while(yes_no != 'no'):

# input('Unesite opis filma: ')
film = 'Eddie, an indomitable Hong Kong cop, is transformed into an immortal warrior with superhuman powers after a fatal accident involving a mysterious medallion. Eddie enlists the help of British Interpol agent Nicole to determine the secret of the medallion and face down the evil Snakehead who wants to use its magical powers for his own nefarious plans.'

film_rep = vec.transform([film])
predicted = clf.predict(film_rep)

res = ''

print(predicted[0, :].toarray()[0])

for genre, prediction in zip(genres, predicted[0, :].toarray()[0]):
    if prediction == 1:
        res += genre + ', '

print(res[:-2])

# yes_no = input('Da li zelite jos filmova [yes/no]: ')
