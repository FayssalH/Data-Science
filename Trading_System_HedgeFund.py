# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:59:10 2023

@author: Fayssal
"""
import yfinance as yf
import quandl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import quandl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pyfolio as pf
import pandas as pd

# Clé API Quandl
quandl.ApiConfig.api_key = 'PLn-UZGsTSy_cVMKWYxG'

# Liste de 10 actions diversifiées
diverse_stocks = ['AAPL', 'MSFT', 'JPM', 'WFC', 'NKE', 'MCD', 'JNJ', 'PFE', 'MMM', 'CAT']

# Télécharger les données historiques pour les actions, le pétrole et les taux d'intérêt de la FED
stock_data = yf.download(diverse_stocks, start='2018-01-01', end='2022-12-31')['Close']
oil_data = yf.download('CL=F', start='2018-01-01', end='2022-12-31')['Close']
fed_rate_data = quandl.get("FRED/FEDFUNDS", start_date='2018-01-01', end_date='2022-12-31')['Value']

# Télécharger les données historiques pour le PIB et le taux de chômage
gdp_data = quandl.get("FRED/GDP", start_date='2018-01-01', end_date='2022-12-31')['Value']
unemployment_rate_data = quandl.get("FRED/UNRATE", start_date='2020-01-01', end_date='2022-12-31')['Value']

# Préparation des données pour le modèle de prédiction
stock_data['GDP'] = gdp_data.reindex(stock_data.index, method='ffill') # Ajouter les données du PIB
stock_data['UnemploymentRate'] = unemployment_rate_data.reindex(stock_data.index, method='ffill') # Ajouter les données du taux de chômage
stock_data.dropna(inplace=True)



# We wanted to create a simple regression model, based on the rates that the FED announce, and the WTI price, which can be a macroeconomic indicator

# Séparation des données en jeu de test et d'entraînement
X_train, X_test, y_train, y_test = train_test_split(stock_data.drop('AAPL', axis=1), stock_data['AAPL'], test_size=0.2, random_state=0)

# Création du modèle de prédiction
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction des prix de clôture
predictions = model.predict(X_test)

# Backtest
returns = pd.Series(predictions, index=y_test.index).pct_change()
pf.create_simple_tear_sheet(returns)


# After seeing that our model is too simple, and our predictions are really bad, we wanted to use ML models, such as Decision Tree and GB Regression:
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Création du modèle de prédiction avec un arbre de décision
model_tree = DecisionTreeRegressor(random_state=0)
model_tree.fit(X_train, y_train)

# Prédiction des prix de clôture avec l'arbre de décision
predictions_tree = model_tree.predict(X_test)

# Backtest pour l'arbre de décision
returns_tree = pd.Series(predictions_tree, index=y_test.index).pct_change()
pf.create_simple_tear_sheet(returns_tree)

# Création du modèle de prédiction avec le gradient boosting
model_gb = GradientBoostingRegressor(random_state=0)
model_gb.fit(X_train, y_train)

# Prédiction des prix de clôture avec le gradient boosting
predictions_gb = model_gb.predict(X_test)

# Backtest pour le gradient boosting
returns_gb = pd.Series(predictions_gb, index=y_test.index).pct_change()
pf.create_simple_tear_sheet(returns_gb)

# Préparation des données pour le modèle de prédiction
features = stock_data.drop('AAPL', axis=1)
target = stock_data['AAPL']

# Séparation des données en jeu de test et d'entraînement
X = features.values
y = target.values

# Initialiser le scaler
scaler = MinMaxScaler()

# Créer un modèle séquentiel
model_dl = Sequential()
model_dl.add(Dense(64, activation='relu', input_dim=features.shape[1]))
model_dl.add(Dense(64, activation='relu'))
model_dl.add(Dense(1))
model_dl.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Initialiser la validation croisée sur une série temporelle
tscv = TimeSeriesSplit(n_splits=5)

# Boucle sur les splits de temps
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Scaling the features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Entraînement du modèle
    model_dl.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

# Prédiction des rendements
predictions_dl = model_dl.predict(X_test)

# Convertir les prédictions en série pandas
predictions_dl = pd.Series(predictions_dl.flatten(), index=stock_data.index[test_index])

# Calculer les rendements
returns_dl = predictions_dl.pct_change()

# Plot des rendements
plt.figure(figsize=(10,6))
plt.plot(returns_dl)
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Returns of Strategy Based on Deep Learning')
plt.show()

# Calculer les rendements quotidiens
daily_returns_dl = returns_dl + 1

# Calculer les rendements cumulés
cumulative_returns_dl = (daily_returns_dl.cumprod() - 1) * 100

# Plot des rendements cumulés
plt.figure(figsize=(10,6))
plt.plot(cumulative_returns_dl)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (%)')
plt.title('Cumulative Returns of Strategy Based on Deep Learning')
plt.show()


import yfinance as yf
import quandl
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Préparation des données pour le modèle de prédiction
stock_data['GDP'] = gdp_data.reindex(stock_data.index, method='ffill') # Ajouter les données du PIB
stock_data['UnemploymentRate'] = unemployment_rate_data.reindex(stock_data.index, method='ffill') # Ajouter les données du taux de chômage
stock_data.dropna(inplace=True)

X = stock_data.drop('AAPL', axis=1)
y = stock_data['AAPL']

tscv = TimeSeriesSplit(n_splits=5)

# Initialiser le scaler
scaler = MinMaxScaler()

# Scaling the features
X = scaler.fit_transform(X)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Boucle sur les splits de temps
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Création du modèle de prédiction avec un réseau de neurones
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Entraînement du modèle
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])
    
# Prédiction des prix de clôture
predictions_nn = model.predict(X_test)

# Convertir les prédictions en série pandas
predictions_nn = pd.Series(predictions_nn.flatten(), index=y_test.index)

# Calculer les rendements
returns_nn = predictions_nn.pct_change()

# Plot des rendements
plt.figure(figsize=(10,6))
plt.plot(returns_nn)
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Returns of Strategy Based on Neural Network')
plt.show()

# Calculer les rendements quotidiens
daily_returns = returns_nn + 1

# Calculer les rendements cumulés
cumulative_returns = (daily_returns.cumprod() - 1) * 100

# Plot des rendements cumulés
plt.figure(figsize=(10,6))
plt.plot(cumulative_returns)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (%)')
plt.title('Cumulative Returns of Strategy Based on Neural Network')
plt.show()

