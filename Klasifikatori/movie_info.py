import pandas as pd
import numpy as np
import time

start_time = time.time()
data_frame = pd.read_csv(
    'Klasifikatori/movies_genres_en.csv', delimiter='\t', index_col=0)
end_time = time.time()

print('Procitao fajl za: ', end_time-start_time, 's')

df_genres = data_frame.drop(['title', 'plot', 'plot_lang'], axis=1)
genres = np.array(data_frame.drop(
    ['plot', 'title', 'plot_lang'], axis=1).columns.values)

print('Broj zanrova: ', len(genres))
print(genres)

counts = []
num_of_movies = len(data_frame)

total_genres_sum = 0

for genre in genres:
    sum_ = df_genres[genre].sum()
    total_genres_sum += sum_
    counts.append((genre, sum_, '{:.3f} %'.format((sum_/num_of_movies)*100)))

print('Prosjecno zanrova po filmu: ', total_genres_sum/num_of_movies)

df_stats = pd.DataFrame(
    counts, columns=['genre', 'number of movies', 'percentage'])
print(df_stats)
