import pandas
import numpy as np

dataset = pandas.read_csv("./data/FirstDataset.csv")

removedDataset = dataset.dropna()

X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=numpy.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
print(X)

from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
print(y)

from sklearn.preprocessing import StandardScaler
norm = StandardScaler()
X = norm.fit_transform(X)
print(X)