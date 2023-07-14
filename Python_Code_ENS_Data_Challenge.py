# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:16:38 2023

@author: Fayssal
"""
"""

Import the csv files, preprocessing the data, clean it and try to get data the most relevant as possible

"""
import math
from statsmodels.formula import api
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Data processing, modeling, and model evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, plot_confusion_matrix

# Randomization
import random

X_train = pd.read_csv('C://Users/Fayssal/Desktop/Neoma Finance & Big Data/Projet Final Python/project_ML_AI/project/X_train_NHkHMNU.csv')
Y_train = pd.read_csv('C://Users/Fayssal/Desktop/Neoma Finance & Big Data/Projet Final Python/project_ML_AI/project/y_train.csv')
X_test = pd.read_csv('C://Users/Fayssal/Desktop/Neoma Finance & Big Data/Projet Final Python/project_ML_AI/project/X_test_final.csv')
Y_test = pd.read_csv('C://Users/Fayssal/Desktop/Neoma Finance & Big Data/Projet Final Python/project_ML_AI/project/y_test_random_final.csv')
X_train.shape
Y_train.shape

X_train.info()
Y_train.info()

X_train.isnull().values.any()
X_train.isna().sum()

X_train = X_train.fillna(0)
X_train = X_train.drop('COUNTRY', axis=1)
Y_train.isna().sum()
Y_train.TARGET.describe()










# %%
#Let us first analyze the distribution of price change 

plt.figure(figsize=[8,4])
sns.distplot(Y_train.TARGET, color='g',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Target Variable Distribution - Daily variation in the price of electricity futures')
plt.show()

#Normality test

from scipy.stats import shapiro

# Perform Shapiro-Wilk test on the target variable
stat, p = shapiro(Y_train.TARGET)
print(f"Shapiro-Wilk test statistic: {stat}")
print(f"Shapiro-Wilk test p-value: {p}")

# Evaluate the null hypothesis
alpha = 0.05
if p > alpha:
    print("The target variable appears to follow a normal distribution (fail to reject H0)")
else:
    print("The target variable does not follow a normal distribution (reject H0)")
    
#The variable doesn't follow a normal distribution


# %%
#Checking number of unique rows in each feature

features = [i for i in X_train.columns]
nu = X_train[features].nunique().sort_values()
nf = []; cf = []; nnf = 0; ncf = 0; #numerical & categorical features

for i in range(X_train[features].shape[1]):
    if nu.values[i]<=16:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

print('\n\033[1mInference:\033[0m The Datset has {} numerical & {} categorical features.'.format(len(nf),len(cf)))

# %% [markdown]
# ##### no categorical features in our dataset 

# %% [markdown]
# ### Data Preprocessing

# %%
# Removal of any Duplicate rows (if any)

print(X_train.shape)
X_train.drop_duplicates(inplace=True)
print(X_train.shape)

# The dataset doesn't have any duplicates

# %%
#Removal of outlier:

df1 = X_train.copy()


for i in nf:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop=True)
print(df1.head())
print('\n\033[1mInference:\033[0m\nBefore removal of outliers, The dataset had {} samples.'.format(X_train.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))

# %% [markdown]
# ### Data Manipulation

# %%
#Feature Scaling (Standardization)

std = StandardScaler()

ID =  X_train.ID
X_train = X_train.drop('ID', axis=1)

print('\033[1mStandardardization on Training set'.center(120))
X_train_std = std.fit_transform(X_train)
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
print(X_train_std.describe())

#print('\n','\033[1mStandardardization on Testing set'.center(120))
#Test_X_std = std.transform(Test_X)
#Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)
#display(Test_X_std.describe())

# %% [markdown]
# ### Feature Selection/Extraction

# %%
X_train.insert(0, 'ID', ID)
df = pd.merge(X_train, Y_train, on="ID")
print(df)

# %%
#Checking the correlation



print('\033[1mCorrelation Matrix'.center(100))
plt.figure(figsize=[25,20])
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, center=0) 
plt.show()

# %% [markdown]
# ##### we can see strong multi-correlation between the features

# %%
Train_xy = pd.concat([X_train_std,Y_train.reset_index(drop=True)],axis=1)
Train_xy.head()

# %%
#Testing a Linear Regression model with statsmodels

Train_xy = pd.concat([X_train_std,Y_train.reset_index(drop=True)],axis=1)
a = Train_xy.columns.values

API = api.ols(formula='{} ~ {}'.format("TARGET",' + '.join(i for i in X_train.columns)), data=Train_xy).fit()
#print(API.conf_int())
#print(API.pvalues)
API.summary()

# %%
# Multiple Linear Regression

MLR = LinearRegression().fit(X_train_std,Y_train)
pred1 = MLR.predict(X_train_std)
pred2 = MLR.predict(X_train_std)

print('{}{}\033[1m Evaluating Multiple Linear Regression Model \033[0m{}{}\n'.format('<'*3,'-'*35 ,'-'*35,'>'*3))
print('The Coeffecient of the Regresion Model was found to be ',MLR.coef_)
print('The Intercept of the Regresion Model was found to be ',MLR.intercept_)

# %% [markdown]
# ##### (Faire cross validation)














    
"""
STEP 2 : Calculate the Spearman Correlation based on a Classical Linear Regression
"""
print('\n','\033[1mStandardardization on Testing set'.center(120))
X_test = X_test.drop('COUNTRY', axis=1)
ID = X_test['ID']
X_test = X_test.drop('ID', axis=1)
Test_X_std = std.fit_transform(X_test)
Test_X_std = pd.DataFrame(Test_X_std, columns=X_test.columns)
print(Test_X_std.describe())
Test_X_std = Test_X_std.fillna(0)

X_test_clean = X_test.fillna(0)
X_test_clean.dropna(inplace=True)

Test_X_std.insert(0, 'ID', ID)

lr = LinearRegression()

X_train_clean = X_train
Y_train_clean = Y_train['TARGET']

#Normalize the data available in X

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_clean)

lr.fit(X_train_clean, Y_train_clean)

output_train = lr.predict(X_train_clean)

def metric_train(output):

    return  spearmanr(output, Y_train_clean).correlation

print("Corrélation (Spearman) pour les données d'entrainement : {:.1f}%".format(100 * metric_train(output_train) ))

output_test_lr = lr.predict(Test_X_std)
Y_test_clean = Y_test['TARGET']

print("Corrélation (Spearman) pour les données de test (Régression linéaire) : {:.1f}%".format(100 * spearmanr(output_test_lr, Y_test_clean).correlation))



























"""
STEP 3 : Calculate Spearman Correlations with other methods, as Neural Network, Random Forest, Lasso ... regressions

Compare them and choose the one who fits the best with our context.
"""


#RandomForest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV
grid_search.fit(X_train_std, Y_train_clean)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters found by GridSearchCV: ", best_params)
print("Best score found by GridSearchCV: {:.1f}%".format(100 * best_score))

# Train the RandomForestRegressor
best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(X_train_std, Y_train_clean)

# Compute the Spearman correlation on the training set
output_train_rf = best_rf.predict(X_train_std)
print("Corrélation (Spearman) pour les données d'entraînement (forêt aléatoire) : {:.1f}%".format(100 * metric_train(output_train_rf)))

#Coefficient of Determination
train_score_rf = best_rf.score(X_train_std, Y_train_clean)
print("R^2 score for training data (Random Forest): {:.1f}%".format(100 * train_score_rf))

output_test_rf = best_rf.predict(Test_X_std)


print("Corrélation (Spearman) pour les données de test (forêt aléatoire) : {:.1f}%".format(100 * spearmanr(output_test_rf, Y_test_clean).correlation))


















#Regression Ridge

from sklearn.linear_model import Ridge

# Define the parameter grid for GridSearchCV
param_grid_ridge = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Create the Ridge object
ridge = Ridge(random_state=42)

# Instantiate the GridSearchCV object
grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid_ridge, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search_ridge.fit(X_train_std, Y_train_clean)

# Get the best parameters and the corresponding score
best_params_ridge = grid_search_ridge.best_params_
best_score_ridge = grid_search_ridge.best_score_

print("Best parameters found by GridSearchCV (Ridge): ", best_params_ridge)
print("Best score found by GridSearchCV (Ridge): {:.1f}%".format(100 * best_score_ridge))

# Train the Ridge model with the best parameters on the whole dataset
best_ridge = Ridge(**best_params_ridge, random_state=42)
best_ridge.fit(X_train_std, Y_train_clean)

# Compute the Spearman correlation on the training set
output_train_ridge = best_ridge.predict(X_train_std)
print("Corrélation (Spearman) pour les données d'entraînement (Ridge) : {:.1f}%".format(100 * metric_train(output_train_ridge)))

train_score_ridge = best_ridge.score(X_train_std, Y_train_clean)
print("R^2 score for training data (Ridge): {:.1f}%".format(100 * train_score_ridge))

#Same with test dataset
output_test_ridge = best_ridge.predict(Test_X_std)
print("Corrélation (Spearman) pour les données de test (Ridge) : {:.1f}%".format(100 * spearmanr(output_test_ridge, Y_test_clean).correlation))
































#Lasso Regression

from sklearn.linear_model import Lasso

# Define the parameter grid for GridSearchCV
param_grid_lasso = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Create the Lasso object
lasso = Lasso(random_state=42)

# Instantiate the GridSearchCV object
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid_lasso, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search_lasso.fit(X_train_std, Y_train_clean)

# Get the best parameters and the corresponding score
best_params_lasso = grid_search_lasso.best_params_
best_score_lasso = grid_search_lasso.best_score_

print("Best parameters found by GridSearchCV (Lasso): ", best_params_lasso)
print("Best score found by GridSearchCV (Lasso): {:.1f}%".format(100 * best_score_lasso))

# Train the Lasso model with the best parameters on the whole dataset
best_lasso = Lasso(**best_params_lasso, random_state=42)
best_lasso.fit(X_train_std, Y_train_clean)

# Compute the Spearman correlation on the training set
output_train_lasso = best_lasso.predict(X_train_std)
print("Corrélation (Spearman) pour les données d'entraînement (Lasso) : {:.1f}%".format(100 * metric_train(output_train_lasso)))

train_score_lasso = best_lasso.score(X_train_std, Y_train_clean)
print("R^2 score for training data (Lasso): {:.1f}%".format(100 * train_score_lasso))


output_test_lasso = best_lasso.predict(Test_X_std)
print("Corrélation (Spearman) pour les données de test (Lasso) : {:.1f}%".format(100 * spearmanr(output_test_lasso, Y_test_clean).correlation))




























#ElasticNet Regression

from sklearn.linear_model import ElasticNet

# Define the parameter grid for GridSearchCV
param_grid_elasticnet = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Create the ElasticNet object
elasticnet = ElasticNet(random_state=42, max_iter = 100000)

# Instantiate the GridSearchCV object
grid_search_elasticnet = GridSearchCV(estimator=elasticnet, param_grid=param_grid_elasticnet, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search_elasticnet.fit(X_train_std, Y_train_clean)

# Get the best parameters and the corresponding score
best_params_elasticnet = grid_search_elasticnet.best_params_
best_score_elasticnet = grid_search_elasticnet.best_score_

print("Best parameters found by GridSearchCV (ElasticNet): ", best_params_elasticnet)
print("Best score found by GridSearchCV (ElasticNet): {:.1f}%".format(100 * best_score_elasticnet))

# Train the ElasticNet model with the best parameters on the whole dataset
best_elasticnet = ElasticNet(**best_params_elasticnet, random_state=42)
best_elasticnet.fit(X_train_std, Y_train_clean)

# Compute the Spearman correlation on the training set
output_train_elasticnet = best_elasticnet.predict(X_train_std)
print("Corrélation (Spearman) pour les données d'entraînement (ElasticNet) : {:.1f}%".format(100 * metric_train(output_train_elasticnet)))

train_score_elasticnet = best_elasticnet.score(X_train_std, Y_train_clean)
print("R^2 score for training data (Lasso): {:.1f}%".format(100 * train_score_elasticnet))

output_test_elasticnet = best_elasticnet.predict(Test_X_std)

print("Corrélation (Spearman) pour les données de test (ElasticNet) : {:.1f}%".format(100 * spearmanr(output_test_elasticnet, Y_test_clean).correlation))




























#Neural network for Spearman Correlation

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Création du modèle
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_clean.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train_clean, Y_train_clean, epochs=100, batch_size=32, verbose=0)

# Prédiction avec le modèle
output_train_ann = model.predict(X_train_clean).reshape(-1)

# Calcul de la corrélation de Spearman
print("Corrélation (Spearman) pour les données d'entraînement (réseau de neurones) : {:.1f}%".format(100 * metric_train(output_train_ann)))

from sklearn.metrics import r2_score

# Compute the R^2 score on the training set for the Neural Network
train_score_ann = r2_score(Y_train_clean, output_train_ann)
print("R^2 score for training data (Neural Network): {:.1f}%".format(100 * train_score_ann))

output_test_ann = model.predict(Test_X_std).reshape(-1)
print("Corrélation (Spearman) pour les données de test (réseau de neurones) : {:.1f}%".format(100 * spearmanr(output_test_ann, Y_test_clean).correlation))


test_score_ann = r2_score(Y_test_clean, output_test_ann)
print("R^2 score for test data (Neural Network): {:.1f}%".format(100 * test_score_ann))





















#PCA for Spearman Correlation Calculation

# Ajout de l'ACP à la fin de l'importation des bibliothèques
from sklearn.decomposition import PCA

# Effectuez l'ACP après le prétraitement des données
X_train_clean = X_train.drop('ID', axis=1)

# Normalisez les données avant d'appliquer l'ACP
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_clean)

# Appliquez l'ACP sur les données normalisées
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)

# Affichez la variance expliquée par chaque composante principale
explained_variance = pca.explained_variance_ratio_
print("Variance expliquée par chaque composante principale:\n", explained_variance)

# Choisissez le nombre de composantes principales en fonction du pourcentage de variance expliquée souhaité
cumulative_variance = np.cumsum(explained_variance)
target_variance = 0.95  # 95% de variance expliquée
n_components = np.where(cumulative_variance >= target_variance)[0][0] + 1

print(f"Nombre de composantes principales pour expliquer {100 * target_variance:.0f}% de la variance: {n_components}")

# Refaites l'ACP avec le nombre de composantes principales sélectionné
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_std)

# Utilisez les données transformées par l'ACP pour entraîner votre modèle
# Par exemple, en utilisant la régression linéaire
lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, Y_train_clean)

# Prédiction avec le modèle utilisant l'ACP
output_train_pca = lr_pca.predict(X_train_pca)

# Calcul de la corrélation de Spearman pour les données d'entraînement
print("Corrélation (Spearman) pour les données d'entraînement avec ACP : {:.1f}%".format(100 * metric_train(output_train_pca)))

train_score_pca = r2_score(Y_train_clean, output_train_pca)
print("R^2 score for training data (PCA): {:.1f}%".format(100 * train_score_pca))




















#Create a function that will plot the correlation circle based on the PCA

import seaborn as sns

def plot_pca_correlation_circle(pca, features, figsize=(10, 10), top_n=8):
    pcs = pca.components_
    
    fig, ax = plt.subplots(figsize=figsize)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False)
    ax.add_artist(circle)
    
    # Distance from the center of the circle
    distances = np.sqrt(pcs[0, :]**2 + pcs[1, :]**2)
    
    # We wanted to just print a certain number of variable in written, because if we don't limit the number, the chart is not clear at all.
    top_n_indices = np.argsort(distances)[-top_n:]
    
    # Circle creation
    for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
        ax.plot([0, x], [0, y], color='k')
        if i in top_n_indices:
            ax.text(x, y, features[i], fontsize=14, color='darkblue', ha='center', va='center')
        ax.scatter(x, y, color='darkblue', marker='o', s=50, edgecolors='k')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)', fontsize=16)
    ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)', fontsize=16)
    ax.set_title('Cercle des corrélations (PCA)', fontsize=18)
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)
    ax.tick_params(axis='both', which='major', labelsize=14)


plot_pca_correlation_circle(pca, X_train_clean.columns, top_n=8)


#Create the Chart of edxplained variance (80%) based on our PCA

X_train_pca = pca.fit_transform(X_train_std)
explained_variance = pca.explained_variance_ratio_

# Find the number of components that explain 80% of the variance
cumulative_variance = np.cumsum(explained_variance)
target_variance = 0.80
n_components = np.where(cumulative_variance >= target_variance)[0][0] + 1
#Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(1, n_components + 1), explained_variance[:n_components], alpha=0.5, align='center', label='Explained variance by component')
ax.step(range(1, n_components + 1), cumulative_variance[:n_components], where='mid', label="Total explained variance")
ax.set_ylabel('Variance Explained')
ax.set_xlabel('Principal Components')
ax.legend(loc='best')
ax.set_title("PCA - Explained Variance by the components")
feature_names = X_train.columns

for i in range(n_components):
    sorted_features = sorted(zip(pca.components_[i], feature_names), key=lambda x: abs(x[0]), reverse=True)
    top_feature_name = sorted_features[0][1]
    ax.text(i + 1, ax.get_ylim()[1] * 0.05, top_feature_name, rotation=45, ha='center', va='baseline', fontsize=12, color='blue')

plt.tight_layout()
plt.show()


























"""
Because we need to use the best model possible to explain the price variation of electricity futures contracts, we will plot a chart
which will show us the Spearman Correlation, and we will take the highest one to deepen our analysis.
"""

# Create a dictionary that will store 
spearman_correlations = {
    'Linear Regression': metric_train(output_train),
    'Random Forest': metric_train(output_train_rf),
    'Ridge': metric_train(output_train_ridge),
    'Lasso': metric_train(output_train_lasso),
    'ElasticNet': metric_train(output_train_elasticnet),
    'Neural Network': metric_train(output_train_ann),
    'PCA': metric_train(output_train_pca)
}

# Dictionary to DF
spearman_correlations_df = pd.DataFrame(list(spearman_correlations.items()), columns=['Method', 'Spearman Correlation'])
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid", font_scale=1.3)
palette = sns.color_palette("husl", len(spearman_correlations))

sns.barplot(x='Method', y='Spearman Correlation', data=spearman_correlations_df, palette=palette)
plt.ylabel('Spearman Correlation')
plt.title('Comparison of the different methods', fontsize=20, fontweight='bold', pad=20)
plt.xticks(rotation=45)
plt.ylim(0, 1)

for index, row in spearman_correlations_df.iterrows():
    plt.text(index, row['Spearman Correlation'] + 0.01, round(row['Spearman Correlation'], 2), color='black', ha='center', fontsize=14)

plt.show()


























"""
Comparison : The best model for our data challenge seems to be the Random Forest method. So, we'll study which variables are really important
to explain the price of futures contracts on the electricity
"""

feature_importances = best_rf.feature_importances_

# We want the top 10 features
sorted_idx = feature_importances.argsort()[-10:][::-1]
sorted_idx = np.clip(sorted_idx, 0, len(X_train_clean.columns) - 1)

sorted_features = X_train_clean.columns[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

# Bar Chart
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Top 10 des variables les plus impactantes (Random Forest)', fontsize=16)
plt.show()

