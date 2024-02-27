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
import pandas as pd



def set_model(num_feat):
    #define neural network
    model_nn = keras.Sequential()
    model_nn.add(layers.Dense(32, activation='relu', input_shape=(num_feat,)))
    model_nn.add(layers.Dense(64, activation='relu'))
    model_nn.add(layers.Dense(2, activation='softmax'))

    return model_nn

def train_model(df,model):
    Y=np.array(df['Etiqueta'])
    #We use the dataframe to define the train and test sets
    x_train,x_test,y_train,y_test= train_test_split(df.drop(columns='Etiqueta'), Y, test_size=0.2, random_state=1, shuffle=False)
    #We use an oversampler to balance our data
    #oversampler = RandomOverSampler()

    #x_train,y_train  = oversampler.fit_resample(x_train,y_train)
    #We we reshape our training set
    x_train=np.array(x_train)
    y_train_encoded=to_categorical(y_train,2)




    #We define a callback to prevent overfitting
    callback = keras.callbacks.ModelCheckpoint(
        filepath=None,
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

# num_features = df.shape[1]
# model=set_model(num_features)
# train_model(df,model).save("models/model.keras")


# folder_path = "path/to/your/folder/"
# model_path = folder_path + "model.keras.h5"
# saved_model = load_model(model_path)
# print(AUC_score_model(saved_model,df))
folder_path = "topSets"
read_subsets_from_folder(folder_path)