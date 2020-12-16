import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

TRAIN_DATA_PATH = "C:/Users/elisa/Desktop/SCHOOL/INF8215/TP/TP3/data/train-minus.csv"
TEST_DATA_PATH = "C:/Users/elisa/Desktop/SCHOOL/INF8215/TP/TP3/data/for-testing.csv"
SUBMISSION_PATH = "C:/Users/elisa/Desktop/SCHOOL/INF8215/TP/TP3/data/results-minus-test.csv"
N_EPOCHS = 175
BATCH_SIZE = 150
LEARNING_RATE = 0.085
VALIDATION_SIZE = 0.15

def create_model():
    model = Sequential(name='DNN')

    model.add(Dense(40, input_dim=87, activation='sigmoid', name='dense_1'))
    model.add(Dense(18, input_dim=40, activation='sigmoid', name='dense_2'))
    #model.add(Dense(2, input_dim=6, activation='sigmoid', name='dense_3'))
    #model.add(Dense(6, input_dim=8, activation='sigmoid', name='dense_4'))
    model.add(Dense(1, input_dim=18, activation='sigmoid', name='dense_3'))

    model.compile(loss='binary_crossentropy',
              optimizer=SGD(learning_rate=LEARNING_RATE),
              metrics=['accuracy'])
    
    return model

def get_training_data():
    dataset = pd.read_csv(TRAIN_DATA_PATH, header=None,  skiprows=1)

    Y = dataset[dataset.columns[-1]].to_numpy()
    X = dataset.drop([0, 88], axis=1).to_numpy()

    for i in range(len(Y)):
        Y[i] = 1 if Y[i] == "phishing" else 0

    return X, Y.astype(np.int)

def get_test_data():
    return pd.read_csv(TEST_DATA_PATH, header=None, skiprows=1).drop([0, 88], axis=1).to_numpy()

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
    print("here")
    data = s.transform(data)
    Y_pred = model.predict(data)
    Y_pred = (Y_pred > 0.5)
    results = [["idx", "status"]]
    for i in range(len(Y_pred)):
        results.append([str(i), "phishing"]) if Y_pred[i][0] == True else results.append([str(i), "legitimate"])

    np.savetxt(SUBMISSION_PATH, np.array(results), delimiter=",", fmt="%s")

def k_fold_cross_validate():
    acc_per_fold = []
    loss_per_fold = []

    X, Y = get_data()
    kf = KFold(n_splits=5)
    fold_no = 1
    for idx, (train_index, val_index) in enumerate(kf.split(X)):

        model = create_model()

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        x_train, x_validation = X[train_index], X[val_index]
        y_train, y_validation = Y[train_index], Y[val_index]
        s = StandardScaler()
        x_train = s.fit_transform(x_train)
        x_validation = s.transform(x_validation)
        history = model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=0, validation_data=(x_validation, y_validation), shuffle=True)
    
        # Generate generalization metrics
        r = model.predict(X, Y, verbose=0)
        r = (r > 0.5)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

      # Increase fold number
        fold_no = fold_no + 1
  

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
      print('------------------------------------------------------------------------')
      print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

#Bagging
#X, Y = get_training_data()
#X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.2)
#bg = BaggingClassifier(Sequential(), n_estimators=20)
#bg.fit(X_train, y_train)
#print(bg.score(X_validation, y_validation))

#test_data = get_test_data()
#Y_pred = bg.predict(test_data)
#Y_pred = (Y_pred > 0.5)
#results = [["idx", "status"]]
#for i in range(len(Y_pred)):
#    results.append([str(i), "phishing"]) if Y_pred[i] == True else results.append([str(i), "legitimate"])

#np.savetxt(SUBMISSION_PATH, np.array(results), delimiter=",", fmt="%s")


#model.save("C:/Users/elisa/Desktop/SCHOOL/INF8215/TP/TP3/models/soumission1")

X, Y = get_training_data()
X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=VALIDATION_SIZE, stratify=Y)
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_validation = s.transform(X_validation)

model = create_model()
history = model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_data=(X_validation, y_validation))

prediction = model.predict(X)
prediction = (prediction > 0.5)
cf = confusion_matrix(Y, prediction)
print(cf)
display_plots(history)

predict(model, get_test_data())