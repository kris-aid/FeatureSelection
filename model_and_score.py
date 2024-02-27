import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
import os
import re
def set_model(num_feat):
    #define neural network
    model_nn = keras.Sequential()
    model_nn.add(layers.Dense(32, activation='relu', input_shape=(num_feat,)))
    model_nn.add(layers.Dense(64, activation='relu'))
    model_nn.add(layers.Dense(2, activation='softmax'))

    return model_nn

def train_model(df,model):
    Y=np.array(df['Etiqueta'])
    X=df.drop(columns='Etiqueta')
    #We use the dataframe to define the train and test sets
    x_train,x_test,y_train,y_test= train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=False)
    #We use an oversampler to balance our data
    #oversampler = RandomOverSampler()

    #x_train,y_train  = oversampler.fit_resample(x_train,y_train)
    #We we reshape our training set
    x_train=np.array(x_train)
    y_train_encoded=to_categorical(y_train,2)




    #We define a callback to prevent overfitting
    callback = keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True,
        save_weights_only=False
    )

    #We compile our model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    #We fit the model
    model.fit( x_train,
                            y_train_encoded,
                        validation_split=0.2,
                        batch_size = 128,
                        epochs = 100,
                        shuffle=True,
                        callbacks=[callback])
    best_model = model if callback.best else None

    return best_model

def AUC_score_model(model,df):
    Y=np.array(df['Etiqueta'])
    x_train,x_test,y_train,y_test= train_test_split(df.drop(columns='Etiqueta'), Y, test_size=0.2, random_state=1, shuffle=False)
    x_test=np.array(x_test) 
    y_test_encoded=to_categorical(y_test,2)
    y_pred = model.predict(x_test)
    # Get the predicted class (0 or 1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Convert the true labels back to single-column format
    y_test_single = np.argmax(y_test_encoded, axis=1)

    # Calculate AUC score
    auc_score = roc_auc_score(y_test_single, y_pred_classes)
    return auc_score

def read_column_names_from_file(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Strip whitespace and split each line into individual words
    column_names = [word.strip() for line in lines for word in line.split()]

    # Remove duplicates while preserving order
    column_names = list(dict.fromkeys(column_names))

    return column_names
def select_columns(column_names, df):
    # Check if all specified columns are present in the DataFrame
    missing_columns = set(column_names) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in DataFrame.")
    
    return df[column_names].copy()

file_path = 'dataset/features_minmax_procesadas.csv'

df = pd.read_csv(file_path)

set_path = 'topSets'

def read_column_numbers(file_path):
    text=""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        text = ''.join(lines)
    subset_dict=extract_column_numbers(text)
    return subset_dict
def extract_column_numbers(subset_text):
    subsets = re.findall(r'Subset \d+: (\d+(?:,\s*\d+)*)', subset_text)
    result = {}
    for i, subset in enumerate(subsets, 1):
        result[i] = [int(x.strip()) for x in subset.split(',')]
    return result
columnes= read_column_numbers("scores.txt")
# print(columnes)

def select_columns_by_position(column_numbers, df):
    # Check if all specified columns are present in the DataFrame
    missing_columns = set(column_numbers) - set(range(len(df.columns)))
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in DataFrame.")
    # add the "Etiqueta" column to the list
    etiqueta_index = df.columns.get_loc("Etiqueta")
    column_numbers.append(etiqueta_index)
    return df.iloc[:, column_numbers].copy()

for subset_number, column_numbers in columnes.items():
    os.makedirs("models", exist_ok=True)
    new_df = select_columns_by_position(column_numbers, df)
    new_df.to_csv(f"models/subset_{subset_number}_df.csv", index=False)
    num_features = new_df.shape[1]
    print(num_features)
    model = set_model(num_features-1)
    train_model(new_df, model).save(f"models/model_subset_{subset_number}.keras")

folder_path = "models"
for file_name in os.listdir(folder_path):
    if file_name.endswith(".keras"):
        saved_model_path = os.path.join(folder_path, file_name)
        print(saved_model_path)
        saved_model = load_model(saved_model_path)
        #retrieve the "subset"+number from the file name
        subset_name = "subset_"+os.path.splitext(file_name)[0].split("_")[2]
        print(subset_name)
        saved_df_path =subset_name + "_df.csv"
        saved_df_path = os.path.join(folder_path, saved_df_path)
        saved_df = pd.read_csv(saved_df_path)
        print(AUC_score_model(saved_model, saved_df))
