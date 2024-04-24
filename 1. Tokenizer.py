import pickle
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import torch
from transformers import AutoTokenizer, AutoModel

# Load the pickled dataset
with open("E:/My projects/NeuralNet_Project/My_dataframe.pkl", "rb") as f:
    Dataset = pickle.load(f)


tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
# model.cuda()  # uncomment it if you have a GPU


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()



print("Начинаю преобразование")
Dataset.insert(
    Dataset.columns.get_loc("Название") + 1,
    "Название_ru_bert_tiny",
    Dataset["Название"].map(lambda x: embed_bert_cls(x, model, tokenizer)),
)
print("Название сделал")
Dataset.insert(
    Dataset.columns.get_loc("Название_комментарий") + 1,
    "Название_комментарий_ru_bert_tiny",
    Dataset["Название_комментарий"].map(
        lambda x: embed_bert_cls(str(x), model, tokenizer)
    ),
)
print("Название_комментарий сделал")
Dataset.insert(
    Dataset.columns.get_loc("Текст законопроекта") + 2,
    "Текст законопроекта_ru_bert_tiny",
    Dataset["Текст законопроекта"].map(
        lambda x: embed_bert_cls(str(x), model, tokenizer)
    ),
)
print("Текст законопроекта сделал")
Dataset.insert(
    Dataset.columns.get_loc("Пояснительная записка") + 2,
    "Пояснительная записка_ru_bert_tiny",
    Dataset["Пояснительная записка"].map(
        lambda x: embed_bert_cls(str(x), model, tokenizer)
    ),
)
print("Пояснительную записку сделал")
Dataset.insert(
    Dataset.columns.get_loc("Финансовое обоснование") + 2,
    "Финансовое обоснование_ru_bert_tiny",
    Dataset["Финансовое обоснование"].map(
        lambda x: embed_bert_cls(str(x), model, tokenizer)
    ),
)
print("Финансовое обоснование сделал")
Dataset.insert(
    Dataset.columns.get_loc("Текст заключений") + 2,
    "Текст заключений_ru_bert_tiny",
    Dataset["Текст заключений"].map(lambda x: embed_bert_cls(str(x), model, tokenizer)),
)
print("Заключение сделал")

print("Задача завершена")

df_num = Dataset
df_num = df_num.drop(
    [
        "Статус закона",
        "Номер",
        "Название",
        "Название_комментарий",
        "Субъект права законодательной инициативы",
        "Предмет ведения",
        "Форма законопроекта",
        "Ответственный комитет",
        "Тематический блок законопроектов",
        "Отрасль законодательства",
        "Профильный комитет",
        "Пояснительная записка",
        "Финансовое обоснование",
        "Текст законопроекта",
        "Текст заключений",
    ],
    axis=1,
)

Dataset.to_pickle("E:/My projects/NeuralNet_Project/My_dataframe.pickle")
df_num.to_pickle("E:/My projects/NeuralNet_Project/My_dataframe_only_data.pickle")

