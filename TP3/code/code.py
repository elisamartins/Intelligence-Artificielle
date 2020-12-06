import numpy as np
import pandas as pd

import matplotlib 
from matplotlib import pyplot as plt

# Téléchargement des données d'entraînement
dataset = pd.read_csv('../data/train.csv', header=None)

# Séparation des features et des trues values (x,y)
Y = dataset[dataset.columns[-1]]
X = dataset.drop(87-1, axis=1)

# Téléchargement dans numpy arra
X = X.to_numpy()
Y = Y.to_numpy()

# Vérification size
print("X:", X.shape)
print("Y:", Y.shape)

# Hyperparamètres
