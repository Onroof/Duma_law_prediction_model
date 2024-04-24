import pickle
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import torch
import re
from openpyxl import Workbook
import openpyxl
from transformers import AutoTokenizer, AutoModel

print('Start')
# Load the pickled dataset
with open("E:/My projects/NeuralNet_Project/My_dataframe_only_data.pickle", "rb") as f:
    Dataset = pickle.load(f)

with open("E:/My projects/NeuralNet_Project/My_dataframe.pickle", "rb") as f:
    Text_len_Dataset = pickle.load(f)

def preparing_dataset():
    global Dataset, Text_len_Dataset
    Dataset = Dataset.replace("None", 0)
    Dataset = Dataset.fillna(0)

    col_name_list = []
    for col in Dataset.columns:
        if 'Отрасль' in col:
            col_name_list.append(col)
    print(col_name_list)

    def replace_russian_text(cell_value):
        russian_pattern = re.compile('[а-яА-Я]+')
        if russian_pattern.search(str(cell_value)):
            return 0
        else:
            return cell_value
    
    for counter in col_name_list:
        Dataset[counter] = Dataset[counter].apply(replace_russian_text)
    
    Dataset = Dataset.replace('0', 0)
    Dataset['Длина_Текст законопроекта'] = Text_len_Dataset['Текст законопроекта'].str.len()

    Dataset['Название_ru_bert_tiny'] = Dataset['Название_ru_bert_tiny'].values.flatten()
    Dataset['Название_комментарий_ru_bert_tiny'] = Dataset['Название_комментарий_ru_bert_tiny'].values.flatten()
    Dataset['Текст законопроекта_ru_bert_tiny'] = Dataset['Текст законопроекта_ru_bert_tiny'].values.flatten()
    Dataset['Пояснительная записка_ru_bert_tiny'] = Dataset['Пояснительная записка_ru_bert_tiny'].values.flatten()
    Dataset['Финансовое обоснование_ru_bert_tiny'] = Dataset['Финансовое обоснование_ru_bert_tiny'].values.flatten()
    Dataset['Текст заключений_ru_bert_tiny'] = Dataset['Текст заключений_ru_bert_tiny'].values.flatten()

    

preparing_dataset()

'''
book = Workbook()
sheet = book.active
book.save('E:/My projects/NeuralNet_Project/My_dataframe_only_data.xlsx')
Dataset.to_excel(
    'E:/My projects/NeuralNet_Project/My_dataframe_only_data.xlsx',
    index=False,
    engine="openpyxl",
)'''
Dataset.to_pickle("E:/My projects/NeuralNet_Project/Preprocessed_data.pickle")
print('finished')