from skrebate import ReliefF
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import random

def average_feature_importance(dataset, n_neighbors, label='Etiqueta', train_test=False):
    # Assuming the label column is named 'label'
    X = dataset.drop(columns=[label], axis=1).values
    y = dataset[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate feature importances for each column and store them
    feature_importances = []
    for column in dataset.drop(label, axis=1).columns:
        fs = ReliefF(n_neighbors=n_neighbors)
        if train_test:
            fs.fit(X_train, y_train)
        else:
            fs.fit(X, y)
        feature_importances.extend(fs.feature_importances_)
    
    # Calculate the average feature importance
    average_importance = np.mean(feature_importances)
    
    return average_importance

# Usage example
dataset = pd.read_csv("dataset/features_minmax_procesadas.csv")
# average_importance = average_feature_importance(dataset, 100)
# print("Average Feature Importance:", average_importance)


def average_feature_importance_subset(dataset, n_neighbors, label='Etiqueta', train_test=False):
    # Randomly select 9 columns (to keep 10 including 'Etiqueta') from the dataset
    subset_columns = random.sample(list(dataset.drop(columns=[label], axis=1).columns), 9)
    subset_columns.append(label)  # Include the 'Etiqueta' column
    subset_dataset = dataset[subset_columns]
    print("Random Subset Columns:")
    print(subset_dataset.columns)
    # Calculate average feature importance on the subset
    average_importance = average_feature_importance(subset_dataset, n_neighbors, label, train_test)

    return average_importance

# Usage example
average_importance_subset = average_feature_importance_subset(dataset, 100)
print("Average Feature Importance (Subset):", average_importance_subset)