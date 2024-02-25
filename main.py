import numpy as np
from skrebate import ReliefF
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
dataset = pd.read_csv("dataset/features_minmax_procesadas.csv")

def objective_function(subset, dataset):
    # Calculate ReliefF score for the given subset of features
    subset_features = dataset.iloc[:, subset]
    score = average_feature_importance(subset_features)
    return score

def average_feature_importance(dataset, n_neighbors=50):
    # Assuming the label column is named 'Etiqueta'
    X = dataset.drop(columns=['Etiqueta'], axis=1).values
    y = dataset['Etiqueta'].values
    fs = ReliefF(n_neighbors=n_neighbors)
    fs.fit(X, y)
    average_importance = np.mean(fs.feature_importances_)
    return average_importance

def initialize_population(num_bats, num_features):
    # Initialize population with random feature subsets
    population = []
    for _ in range(num_bats):
        subset = np.random.choice(num_features, size=np.random.randint(1, num_features+1), replace=False)
        population.append(subset)
    print("Initial Population:", population)
    return population

def update_bat_position(current_position, best_position, alpha, gamma):
    # Update bat position using echolocation
    new_position = current_position + alpha * (best_position - current_position) + gamma * np.random.uniform(-1, 1, len(current_position))
    return new_position.astype(int)

def bat_algorithm(dataset, num_iterations=100, num_bats=2, alpha=0.5, gamma=0.5):
    num_features = len(dataset.columns) - 1  # Exclude the target column ('Etiqueta')
    population = initialize_population(num_bats, num_features)
    best_bat = None
    best_fitness = float('-inf')
    
    for _ in range(num_iterations):
        for i, bat in enumerate(population):
            fitness = objective_function(bat, dataset)
            if fitness > best_fitness:
                best_fitness = fitness
                best_bat = bat
        
            # Update bat position
            new_bat = update_bat_position(bat, best_bat, alpha, gamma)
            population[i] = new_bat
        
    return best_bat, best_fitness

best_subsets = []
for _ in range(5):
    best_subset, best_fitness = bat_algorithm(dataset)
    best_subsets.append((best_subset, best_fitness))

# Sort subsets by fitness
best_subsets.sort(key=lambda x: x[1], reverse=True)

print("Top 5 subsets of features:")
for i, (subset, fitness) in enumerate(best_subsets, start=1):
    print(f"Subset {i}: {subset} - ReliefF Score: {fitness}")
