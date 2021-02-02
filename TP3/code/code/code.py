import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.callbacks import EarlyStopping
import os

TRAIN_DATA_PATH = os.path.join("data/train.csv")
TEST_DATA_PATH = os.path.join("data/test.csv")
SUBMISSION_PATH = os.path.join("data/submission.csv")

N_EPOCHS = 2000
BATCH_SIZE = 128
LEARNING_RATE = 0.02
VALIDATION_SIZE = 0.15
TOTAL_FEATURES = 87
FEATURE_SCORE_THRESHOLD = 1
PATIENCE = 30

def create_model(n_features):
    model = Sequential(name='DNN')

    model.add(Dense(59, input_dim=n_features, activation='sigmoid', name='dense_1'))
    model.add(Dense(10, input_dim=59, activation='sigmoid', name='dense_2'))
    model.add(Dense(1, input_dim=10, activation='sigmoid', name='dense_3'))
  

    model.compile(loss='binary_crossentropy',
              optimizer=SGD(learning_rate=LEARNING_RATE),
              metrics=['accuracy'])
    
    return model

def get_training_data(features):
    dataset = pd.read_csv(TRAIN_DATA_PATH)
    for f in features:
        dataset = dataset.drop([f], axis=1)
    Y = dataset[dataset.columns[-1]].to_numpy()
    X = dataset.drop(dataset.columns[len(dataset.columns) - 1], axis=1).drop(dataset.columns[0], axis=1).to_numpy()

    for i in range(len(Y)):
        Y[i] = 1 if Y[i] == "phishing" else 0

    return X, Y.astype(np.int)

def get_test_data(features):
    dataset = pd.read_csv(TEST_DATA_PATH)
    for f in features:
        dataset = dataset.drop([f], axis=1)
    return dataset.drop(dataset.columns[0], axis=1).to_numpy()

def get_features_to_drop():
    if FEATURE_SCORE_THRESHOLD <= 0:
        return []

    data = pd.read_csv(TRAIN_DATA_PATH)
    Y = data[data.columns[-1]]

    X = data.drop(data.columns[88], axis=1).drop(data.columns[0], axis=1)
    X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X),columns = X.columns)
    Y[Y=="phishing"] = 1
    Y[Y=="legitimate"] = 0
    Y = Y.astype('int')

    best_features = SelectKBest(score_func=chi2, k=TOTAL_FEATURES)
    scores = pd.DataFrame(best_features.fit(X_scaled,Y).scores_)
    features = pd.DataFrame(X_scaled.columns)

    featureScores = pd.concat([features,scores], axis=1)
    featureScores.columns = ['Specs','Score']
    
    return np.delete(featureScores[featureScores.Score < FEATURE_SCORE_THRESHOLD].drop(featureScores.columns[1], axis=1).to_numpy(), 0)


def display_plots(history):
    size = len(history.history['loss'])

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training curve')
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')
    plt.axis([0, size, 0, 1])
    plt.show()

    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Training curve')
    plt.ylabel('Accuracy')
    plt.xlabel('No. epoch')
    plt.axis([0, size, 0, 1])
    plt.show()

def predict(model, data):
    data = s.transform(data)
    Y_pred = model.predict(data)
    Y_pred = (Y_pred > 0.5)
    results = [["idx", "status"]]
    for i in range(len(Y_pred)):
        results.append([str(i), "phishing"]) if Y_pred[i][0] == True else results.append([str(i), "legitimate"])

    np.savetxt(SUBMISSION_PATH, np.array(results), delimiter=",", fmt="%s")


features_to_drop = get_features_to_drop()

X, Y = get_training_data(features_to_drop)
X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=VALIDATION_SIZE, stratify=Y)
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_validation = s.transform(X_validation)

callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE) ]
model = create_model(TOTAL_FEATURES - len(features_to_drop))
history = model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_data=(X_validation, y_validation), callbacks=callbacks)

prediction = model.predict(X)
prediction = (prediction > 0.5)
cf = confusion_matrix(Y, prediction)
print(cf)
display_plots(history)

predict(model, get_test_data(features_to_drop))
