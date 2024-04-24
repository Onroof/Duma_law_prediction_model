# Data Processing
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
from openpyxl import Workbook
import openpyxl

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import torch
import torchmetrics
from transformers import BertModel, BertTokenizer


from scipy.stats import randint
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# Tree Visualisation

# print(pd.__version__)


def my_func(my_string):
    return np.fromstring(my_string.strip('[]'), sep=' ')


print('Start')
# Load the pickled dataset

df = pd.read_pickle(
    'C:/Users/ProgK/Desktop/My projects/Neural net/Preprocessed_data_wo_rassmot.pickle')
df = df.fillna(0)

Dataset = df.loc[df['Созыв'].isin([5, 6, 7, 8])].copy()
# Dataset = df

print('Распределение законов: ', Dataset.value_counts("Принят"))


array_features = np.concatenate([Dataset[col].values.tolist() for col in ['Название_ru_bert_tiny',
                                                                          'Название_комментарий_ru_bert_tiny',
                                                                          'Текст законопроекта_ru_bert_tiny',
                                                                          'Пояснительная записка_ru_bert_tiny',
                                                                          'Финансовое обоснование_ru_bert_tiny',
                                                                          'Текст заключений_ru_bert_tiny']],
                                axis=1)
'''
array_features = np.concatenate([Dataset[col].values.tolist() for col in ['Текст заключений_ru_bert_tiny']],
                                axis=1)
'''
float_features = Dataset.drop(['Принят',
                               'Название_ru_bert_tiny',
                               'Название_комментарий_ru_bert_tiny',
                               'Текст законопроекта_ru_bert_tiny',
                               'Пояснительная записка_ru_bert_tiny',
                               'Финансовое обоснование_ru_bert_tiny',
                               'Текст заключений_ru_bert_tiny',
                               'Дата регистрации',
                               'Время регистрации',
                               'Дата_1_чтение',
                               'Дата_2_чтение',
                               'Дата_3_чтение',
                               'Дата_СФ',
                               'Дата_Президент',
                               'Дата_повт_СФ',
                               'Дата_повт_Пр',
                               'Дата_опубл',

                               '1 чтение',
                               '2 чтение',
                               '3 чтение',
                               'Совет федерации',
                               'Президент',
                               'Опубликование',
                               'Дата_2_чтение',
                               'День_2_чтение',
                               'Месяц_2_чтение',
                               'Год_2_чтение',
                               'Дата_3_чтение',
                               'День_3_чтение',
                               'Месяц_3_чтение',
                               'Год_3_чтение',
                               'Дата_СФ',
                               'День_СФ',
                               'Месяц_СФ',
                               'Год_СФ',
                               'Дата_Президент',
                               'День_Президент',
                               'Месяц_Президент',
                               'Год_Президент',
                               'Дата_опубл',
                               'День_опубл',
                               'Месяц_опубл',
                               'Год_опубл',
                               'Дата_повт_СФ',
                               'День_повт_СФ',
                               'Месяц_повт_СФ',
                               'Год_повт_СФ',
                               'Дата_повт_Пр',
                               'День_повт_Пр',
                               'Месяц_повт_Пр',
                               'Год_повт_Пр'
                               ],
                              axis=1).values

'''
Dataset_values = Dataset.drop(['Принят',
                               'Название_ru_bert_tiny',
                               'Название_комментарий_ru_bert_tiny',
                               'Текст законопроекта_ru_bert_tiny',
                               'Пояснительная записка_ru_bert_tiny',
                               'Финансовое обоснование_ru_bert_tiny',
                               'Текст заключений_ru_bert_tiny',
                               'Дата регистрации',
                               'Время регистрации',
                               'Дата_1_чтение',
                               'Дата_2_чтение',
                               'Дата_3_чтение',
                               'Дата_СФ',
                               'Дата_Президент',
                               'Дата_повт_СФ',
                               'Дата_повт_Пр',
                               'Дата_опубл',

                               '1 чтение',
                               '2 чтение',
                               '3 чтение',
                               'Совет федерации',
                               'Президент',
                               'Опубликование',
                               'Дата_2_чтение',
                               'День_2_чтение',
                               'Месяц_2_чтение',
                               'Год_2_чтение',
                               'Дата_3_чтение',
                               'День_3_чтение',
                               'Месяц_3_чтение',
                               'Год_3_чтение',
                               'Дата_СФ',
                               'День_СФ',
                               'Месяц_СФ',
                               'Год_СФ',
                               'Дата_Президент',
                               'День_Президент',
                               'Месяц_Президент',
                               'Год_Президент',
                               'Дата_опубл',
                               'День_опубл',
                               'Месяц_опубл',
                               'Год_опубл',
                               'Дата_повт_СФ',
                               'День_повт_СФ',
                               'Месяц_повт_СФ',
                               'Год_повт_СФ',
                               'Дата_повт_Пр',
                               'День_повт_Пр',
                               'Месяц_повт_Пр',
                               'Год_повт_Пр'
                               ],
                              axis=1)
print(Dataset_values.columns)
'''

float_features = float_features.astype(float)

X = np.hstack((float_features, array_features))
# X = float_features
# X = array_features
y = Dataset['Принят'].values


print('X shape:', X.shape)
print('y shape:', y.shape)

print(type(X[0][0]))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=33, test_size=0.25)


# --------------------------------
# прогнозирование времени принятия


'''
# --------------------------------
# Random forest
rf_model = RandomForestClassifier(
    n_estimators=10, max_features="auto", random_state=33)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print('Результаты для random forest')
f1_scoring = f1_score(y_test, y_pred, average='macro')
print("f1_scoring_macro:", f1_scoring)
f1_scoring = f1_score(y_test, y_pred, average='weighted')
print("f1_scoring_weighted:", f1_scoring)

scores = cross_val_score(rf_model, X, y, cv=5, scoring='f1_macro')
print("cross_val_score:", scores)
accuracy = balanced_accuracy_score(y_test, y_pred)  # accuracy (доп)
print("balanced_accuracy_score:", accuracy)
Roc_pred = roc_auc_score(y, rf_model.predict_proba(X)[:, 1])  # roc auc (доп)
print("roc_auc_score:", Roc_pred)
# --------------------------------
logreg = LogisticRegression(random_state=16, max_iter=40000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Результаты для лог регрессии')
f1_scoring = f1_score(y_test, y_pred, average='macro')
print("f1_scoring_macro:", f1_scoring)
f1_scoring = f1_score(y_test, y_pred, average='weighted')
print("f1_scoring_weighted:", f1_scoring)

scores = cross_val_score(logreg, X, y, cv=5, scoring='f1_macro')
print("cross_val_score:", scores)
accuracy = balanced_accuracy_score(y_test, y_pred)  # accuracy (доп)
print("balanced_accuracy_score:", accuracy)
Roc_pred = roc_auc_score(y, logreg.predict_proba(X)[:, 1])  # roc auc (доп)
print("roc_auc_score:", Roc_pred)
print('---------------------------------------')'''


'''
list_importances = []
for i in rf_model.feature_importances_:
    list_importances.append(i)

print('list_importances :64')
print(list_importances[:64])
print('sorted list_importances :64')
sorted_list = sorted(list_importances[:64], reverse=True)
print(sorted_list)
'''

print('finished')
