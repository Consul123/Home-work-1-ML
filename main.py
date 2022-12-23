from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import re

app = FastAPI()
df_train = pd.read_csv('train_set.csv')
df_train.drop(['Unnamed: 0'], axis=1, inplace=True)
with open('model_hw1.pkl', 'rb') as f:
    predictor = pickle.load(f)

def reg_clear(x):
    if x == 'nan' or x == ' bhp':
        v = 'nan'
    else:
        if '.' in x:
            result = re.search(r'\d+\.\d*', str(x))
        else:
            result = re.search(r'\d+', str(x))
        v = result.group(0)
    return v

def columns_reg(df_test):
    df_test['mileage'] = df_test['mileage'].astype(str)
    df_test['mileage'] = df_test.mileage.apply(lambda x: reg_clear(x))
    df_test['mileage'] = df_test['mileage'].astype(float)

    df_test['max_power'] = df_test['max_power'].astype(str)
    df_test['max_power'] = df_test.max_power.apply(lambda x: reg_clear(x))
    df_test['max_power'] = df_test['max_power'].astype(float)

    df_test['engine'] = df_test['engine'].astype(str)
    df_test['engine'] = df_test.engine.apply(lambda x: reg_clear(x))
    df_test['engine'] = df_test['engine'].astype(float)

    return df_test

def feat_eng(df_test, df_train):
    mod_engine = df_train['engine'].mode()
    mod_max_power = df_train['max_power'].mode()
    mod_mileage = df_train['mileage'].mode()
    mod_seats = df_train['seats'].mode()

    df_test['engine'] = df_test['engine'].fillna(value=mod_engine[0])
    df_test['max_power'] = df_test['max_power'].fillna(value=mod_max_power[0])
    df_test['mileage'] = df_test['mileage'].fillna(value=mod_mileage[0])
    df_test['seats'] = df_test['seats'].fillna(value=mod_seats[0])



    df_test['seats'] = df_test['seats'].astype(int)
    df_test['hpl'] = df_test['max_power'] / df_test['engine'] * 1000
    df_test['year2'] = df_test['year'] * df_test['year']

    conditions2 = [(((df_test['owner'] == 'First Owner') | (df_test['owner'] == 'Second Owner')) &
                    (df_test['seller_type'] == 'Trustmark Dealer')),
                   (df_test['owner'] == 'Third Owner'),
                   (df_test['owner'] == 'Fourth & Above Owner'),
                   (df_test['seller_type'] == 'Dealer'),
                   (df_test['seller_type'] == 'Individual')]

    values = [1, 0, 0, 0, 0]
    df_test['Good_seller'] = np.select(conditions2, values)

    return df_test

def encode_scale(df_test, df_train):
    big = pd.concat([df_test, df_train])
    big = pd.get_dummies(big, drop_first=True)
    scal = StandardScaler()
    df_fin_test = big.head(df_test.shape[0])
    df_fin_train = big.tail(big.shape[0] - df_test.shape[0])
    scal.fit(df_fin_train)
    df_test = pd.DataFrame(scal.transform(df_fin_test), columns=df_fin_test.columns)
    return df_test

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    cars = dict(item)


    df_pr = pd.DataFrame([item])
    df_pr = df_pr[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']]


    df_pr = columns_reg(df_pr)
    df_pr = feat_eng(df_pr, df_train)
    df_pr = encode_scale(df_pr, df_train)
    score = predictor.predict(df_pr)[0]

    return score.tolist()


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    cars = []
    col = list(dict(list(items)[0]).keys())


    for item in items:
        cars.append(list(dict(item).values()))
    df_pr = pd.DataFrame(cars, columns=col)
    df_pr = df_pr[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']]


    df_pr = columns_reg(df_pr)
    df_pr = feat_eng(df_pr, df_train)
    df_pr = encode_scale(df_pr, df_train)
    scores = predictor.predict(df_pr)
    return scores.tolist()