# Data Processing
import pickle
import pandas as pd
import numpy as np
import os
import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score

print('Start')
Dataframe = pd.read_pickle('E:/My projects/NeuralNet_Project/Preprocessed_data_wo_rassmot_w_date.pickle')
df_base = Dataframe.loc[Dataframe['Созыв'].isin([5, 6, 7, 8])].copy()
'''
array_features = np.concatenate([df_base[col].values.tolist() for col in ['Название_ru_bert_tiny',
                                                                          'Название_комментарий_ru_bert_tiny',
                                                                          'Текст законопроекта_ru_bert_tiny',
                                                                          'Пояснительная записка_ru_bert_tiny',
                                                                          'Финансовое обоснование_ru_bert_tiny',
                                                                          'Текст заключений_ru_bert_tiny']],
                                axis=1)
'''
array_features = np.concatenate([df_base[col].values.tolist() for col in ['Текст заключений_ru_bert_tiny']],
                                axis=1)

float_features = df_base.drop(['Принят',
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


float_features = float_features.astype(float)




#X = np.hstack((float_features, array_features)).astype(np.float32)
#X = float_features.astype(np.float32)
X = array_features.astype(np.float32)
y = df_base['Принят'].values.astype(np.float32)


# разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train.reshape(-1, 1))
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test.reshape(-1, 1))


# создание объектов типа Dataset с помощью класса TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# создание объектов типа DataLoader для загрузки данных батчами
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# определение модели нейронной сети
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.dropout1(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.sigmoid(x)
        return x

model = Net()

# определение функции потерь и оптимизатора
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# обучение модели
for epoch in range(1000):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data

    # обнуление градиентов
    optimizer.zero_grad()

    # прямой проход
    
    outputs = model(inputs)

    # вычисление функции потерь
    loss = criterion(outputs, labels)

    # обратный проход и оптимизация
    loss.backward()
    optimizer.step()

    # добавление текущей потери в running loss
    running_loss += loss.item()

# вычисление метрик на тестовой выборке после каждой эпохи
y_pred = []
y_true = []
with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data
        outputs = model(inputs)
        y_pred += outputs.cpu().numpy().tolist()
        y_true += labels.cpu().numpy().tolist()

f1_weighted = f1_score(y_true, np.round(y_pred), average='weighted')
f1_macro = f1_score(y_true, np.round(y_pred), average='macro')
roc_auc = roc_auc_score(y_true, y_pred)
balanced_accuracy = balanced_accuracy_score(y_true, np.round(y_pred))

print('Epoch: %d | Loss: %.3f | F1 macro: %.3f | F1 weight: %.3f | Bal. acc.: %.3f| roc_auc: %.3f' %
      (epoch + 1, running_loss / len(train_dataloader), f1_macro, f1_weighted, balanced_accuracy, roc_auc))


print('finished')
