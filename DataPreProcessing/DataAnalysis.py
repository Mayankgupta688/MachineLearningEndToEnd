import pandas
import numpy as np

dataset = pandas.read_csv("./data/FirstDataset.csv")

removedDataset = dataset.dropna()

X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import StandardScaler
norm = StandardScaler()
X = norm.fit_transform(X)
print(X)