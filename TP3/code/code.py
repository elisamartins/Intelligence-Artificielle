import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.callbacks import EarlyStopping

TRAIN_DATA_PATH = "C:/Users/elisa/Desktop/AUTOMNE 2020/INF8215/TP/TP3/data/train.csv"
TEST_DATA_PATH = "C:/Users/elisa/Desktop/AUTOMNE 2020/INF8215/TP/TP3/data/test.csv"
SUBMISSION_PATH = "C:/Users/elisa/Desktop/AUTOMNE 2020/INF8215/TP/TP3/data/submission.csv"
N_EPOCHS = 2000
BATCH_SIZE = 128
LEARNING_RATE = 0.02
VALIDATION_SIZE = 0.2
TOTAL_FEATURES = 87
FEATURE_SCORE_THRESHOLD = 20
PATIENCE = 10

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

# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
def get_features_to_drop():
    data = pd.read_csv("C:/Users/elisa/Desktop/AUTOMNE 2020/INF8215/TP/TP3/data/train-minus.csv")
    Y = data[data.columns[-1]]
    #X = data[data.columns[-88:0]]
    data = data.drop(data.columns[88], axis=1).drop(data.columns[0], axis=1)
    X = data
    X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X),columns = X.columns)
    Y[Y=="phishing"] = 1
    Y[Y=="legitimate"] = 0
    Y = Y.astype('int')
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=TOTAL_FEATURES)
    fit = bestfeatures.fit(X_scaled,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_scaled.columns)

    ##concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    f = featureScores[featureScores.Score < FEATURE_SCORE_THRESHOLD].drop(featureScores.columns[1], axis=1).to_numpy()
    f = np.delete(f, 0)
    
    return f


def display_plots(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training curve')
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')
    plt.axis([0, N_EPOCHS, 0, 1])
    plt.show()

    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Training curve')
    plt.ylabel('Accuracy')
    plt.xlabel('No. epoch')
    plt.axis([0, N_EPOCHS, 0, 1])
    plt.show()

def predict(model, data):
    data = s.transform(data)
    Y_pred = model.predict(data)
    Y_pred = (Y_pred > 0.5)
    results = [["idx", "status"]]
    for i in range(len(Y_pred)):
        results.append([str(i), "phishing"]) if Y_pred[i][0] == True else results.append([str(i), "legitimate"])

    np.savetxt(SUBMISSION_PATH, np.array(results), delimiter=",", fmt="%s")

#np.savetxt(SUBMISSION_PATH, np.array(results), delimiter=",", fmt="%s")


#model.save("C:/Users/elisa/Desktop/SCHOOL/INF8215/TP/TP3/models/soumission1")

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
