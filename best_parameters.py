import pandas as pd

results = pd.read_csv('grid_results.csv', delimiter=';')
#results.drop(['score'], axis=1, inplace=True)
results.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

# for each run, get the parameters with lowest silhouette score y
df = results.groupby(['no_components', 'perplexity', 'cluster', 'learning']).mean()
print(df.sort_values('silhouette').head(5))
print(df.sort_values('score').head(5))