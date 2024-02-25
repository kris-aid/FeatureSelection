from skrebate import ReliefF
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
#example of 2 class problem

def top_N_columns_names(dataset, N_columns,n_neighbors,label='Etiqueta',train_test=False):
    # Assuming the label column is named 'label'
    X = dataset.drop(columns=[label],axis=1).values
    y = dataset[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    fs = ReliefF(n_neighbors=n_neighbors, n_features_to_select=N_columns)
    if train_test:
        fs.fit(X_train, y_train)
    else:
        fs.fit(X, y)
    return dataset.columns[np.argsort(fs.feature_importances_)[::-1][:N_columns]]


# Load your dataset
dataset = pd.read_csv("dataset/features_minmax_procesadas.csv")

columns=top_N_columns_names(dataset,5,100)
print(columns)

# # Assuming the label column is named 'label'
# X = dataset.drop(columns=['Etiqueta'],axis=1).values
# y = dataset['Etiqueta'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# fs = ReliefF(n_neighbors=10, n_features_to_select=5)
# fs.fit(X_train, y_train)
# X_with_bestfeaturetes=fs.fit_transform(X_train, y_train)
# print(X_with_bestfeaturetes)
# for feature_name, feature_score in zip(dataset.drop('Etiqueta', axis=1).columns,
#                                        fs.feature_importances_):
#     print(feature_name, '\t', feature_score)
