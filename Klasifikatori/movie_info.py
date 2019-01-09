import pandas as pd
import numpy as np
import time

start_time = time.time()
data_frame = pd.read_csv('movies_genres_en.csv', delimiter='\t')
end_time = time.time()

print('Procitao fajl za: ', end_time-start_time, 's')

df_genres = data_frame.drop(['title', 'plot', 'plot_lang'], axis=1)
genres = np.array(data_frame.drop(['plot', 'title', 'plot_lang'], axis=1).columns.values)

print('Broj zanrova: ', len(genres))
print(genres)

counts = []
num_of_movies = len(data_frame)

for genre in genres:
    sum_ = df_genres[genre].sum()
    counts.append((genre, sum_, '{:.3f} %'.format((sum_/num_of_movies)*100)))

df_stats = pd.DataFrame(counts, columns=['genre', 'number of movies', 'percentage'])
print(df_stats)