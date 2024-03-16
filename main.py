import random
import numpy as np
from skrebate import ReliefF
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Load dataset
dataset = pd.read_csv("dataset/features_minmax_procesadas.csv")
etiqueta_column_number = dataset.columns.get_loc('Etiqueta')

#takes a subset of features and the dataset, calculates the average feature 
#importance using the ReliefF algorithm, and returns the score.
def objective_function(subset, dataset):    
    # Check if etiqueta_column_number is in the subset
    if etiqueta_column_number not in subset:
        # Add etiqueta_column_number to the subset
        subset = np.append(subset, etiqueta_column_number)
    
    subset_features = dataset.iloc[:, subset]
    score = average_feature_importance(subset_features)
    return score

# Relieff algorithm
def average_feature_importance(dataset, n_neighbors=50):
    # print("Calculating feature importances for dataset...", dataset.columns)
    # Assuming the label column is named 'Etiqueta'
    X = dataset.drop(columns=['Etiqueta'], axis=1).values
    y = dataset['Etiqueta'].values
    fs = ReliefF(n_neighbors=n_neighbors)
    fs.fit(X, y)
    average_importance = np.mean(fs.feature_importances_)
    # print("Average Feature Importance:", average_importance)
    return average_importance

#initializes the population with random feature subsets.
def initialize_population(num_bats, num_features, subset_size=10):
    # Initialize population with random feature subsets
    population = []
    for _ in range(num_bats):
        # Randomly select 9 other features that are not etiqueta_column_number
        available_features = [i for i in range(num_features) if i != etiqueta_column_number]
        subset = np.random.choice(available_features, size=subset_size, replace=False)
        population.append(subset)
    # print("Initial Population:", population)
    # Each bat is represented as a list of feature indices ex: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] where each index represents a feature.
    return population

# updates the bat's position using echolocation.
#Exploration:
#The bat's position is updated in such a way that it moves towards the best position found so far 
#(best_position) with a certain step size controlled by alpha.
#Additionally, randomness is introduced to the movement through the gamma parameter,
# allowing for exploration of the search space beyond just moving towards the best solution found.
def update_bat_position(current_position, best_position, alpha, gamma):
    #alpha controls the amplitude of the bat's movement towards the best position.
    #gamma introduces randomness to the bat's movement.
    new_position = current_position + alpha * (best_position - current_position) + gamma * np.random.uniform(-1, 1, len(current_position))
    return new_position.astype(int)

#  main function implementing the Bat Algorithm. It initializes a population, 
# iterates over a specified number of iterations, updates bat positions, and selects 
# the best bat (feature subset) based on the objective function.
def bat_algorithm(dataset, num_iterations=2, num_bats=10, subset_size=70, alpha=0.5, gamma=0.5):
    print("Running Bat Algorithm...")
    num_features = len(dataset.columns) - 1  # Exclude the target column ('Etiqueta')
    population = initialize_population(num_bats, num_features, subset_size)
    best_bat = None
    best_fitness = float('-inf')
    
    for _ in range(num_iterations):
        print("Iteration:", _)
        for i, bat in enumerate(population):
            fitness = objective_function(bat, dataset)
            if fitness > best_fitness:
                best_fitness = fitness
                best_bat = bat
        
            # Update bat position
            new_bat = update_bat_position(bat, best_bat, alpha, gamma)
            population[i] = new_bat
        
    return best_bat, best_fitness

# Usage
iterations = 100
bat_number = 5 
best_subsets = []
selected_features = []
for _ in range(5):
    # Generate random subset size between 25 and 45
    subset_size = random.randint(25, 45)
    selected_features.append(subset_size)
    print(f"\nRunning Bat Algorithm with subset size: {subset_size}")
    best_subset, best_fitness = bat_algorithm(dataset, iterations, bat_number, subset_size)
    best_subsets.append((best_subset, best_fitness))

# Sort subsets by fitness
# best_subsets.sort(key=lambda x: x[1], reverse=True)

# Print and save top subsets
print("\nTop 5 subsets of features:")
for i, (subset, fitness) in enumerate(best_subsets, start=1):
    print(f"Subset {i}: {subset} - ReliefF Score: {fitness}")

# Save top subsets to a file
with open('scores.txt', 'w') as f:
    f.write("Top 5 subsets of features:\n")
    for i, (subset, fitness) in enumerate(best_subsets, start=1):
        subset_str = ', '.join(map(str, subset))
        line = f"Subset {i}: {subset_str} - ReliefF Score: {fitness}\n"
        f.write(line)

print("Output has been saved to 'scores.txt'")

# Get column names
column_names = dataset.columns.tolist()

# Print top subsets with corresponding names
print("\nTop 5 subsets of features with corresponding names:")
for i, (subset, fitness) in enumerate(best_subsets, start=1):
    subset_names = [column_names[i] for i in subset]
    print(f"Subset {i}: {subset_names} - ReliefF Score: {fitness}")

# Create a folder and save subsets in separate files
folder_name = "topSets_multiple_sizes"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for i, (subset, _) in enumerate(best_subsets, start=1):
    subset_names = [column_names[i] for i in subset]
    file_path = os.path.join(folder_name, f"subset_{i}_{selected_features[i-1]}.txt")
    with open(file_path, "w") as file:
        file.write("\n".join(subset_names))

print(f"\nSubsets saved in the '{folder_name}' folder.")