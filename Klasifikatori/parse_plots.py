# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Klasifikatori/movies_genres_en.csv',
                 delimiter='\t', index_col=0)
df.info()

# %% [markdown]
# Potrebno je izracunati apsolutan broj filmova po zanru. Napomena: kako jedan film moze
# pripadati vise zanrova, zbir svih filmova po zanrovima ce biti veci od samog broja filmova

# %%
df_genres = df.drop(['plot', 'title', 'plot_lang'], axis=1)
counts = []
to_drop = []
categories = list(df_genres.columns.values)
for i in categories:
    sum_ = df_genres[i].sum()
    counts.append((i, sum_))
    if sum_ < 1000:
        to_drop.append(i)
df_stats = pd.DataFrame(counts, columns=['genre', '#movies'])
df_stats

# %%
df_stats.plot(x='genre', y='#movies', kind='bar',
              legend=False, grid=True, figsize=(15, 8))

# %% [markdown]
# Neki zanrovi imaju premalo vrijednosti, pa da olaksam racunanje, izbacicu ih
# Izbacio sam sve sa manje 1000 filmova, tj. manje od ~600

# %%
for genre in to_drop:
    df.drop(genre, axis=1, inplace=True)
    print(genre)

# %% [markdown]
# Mozemo sad ostale filmove prepisati u CSV

# %%
df.to_csv('Klasifikatori/movies_genres_en.csv', sep='\t', encoding='utf-8')
