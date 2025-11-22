import pandas as pd

# Cargar datos
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Ver las primeras filas
print(train.head())

# Info general
print(train.info())
