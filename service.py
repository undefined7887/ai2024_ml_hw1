import io
import pickle
import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, Response
from pydantic import BaseModel
from typing import List

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


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


def remove_units(df, columns):
    df_res = df.copy()

    for col in columns:
        if df_res[col].dtype == 'object':
            df_res[col] = df_res[col].str.split(' ').str.get(0)

        def cast_to_float(x):
            if type(x) == str and len(x) > 0:
                return float(x)
            elif type(x) == float:
                return float(x)
            elif type(x) == int:
                return float(x)
            else:
                return np.nan

        df_res[col] = df_res[col].apply(cast_to_float)

    return df_res


def process_df(df):
    # Remove target column, and unnecessary columns
    df = df.drop(columns=['selling_price', 'torque', 'name'], errors='ignore', axis=1)

    numeric_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']

    # Remove units from numeric columns
    df = remove_units(df, columns=numeric_columns)

    # Fill missing values
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Scaling
    df = process_scaler(df, numeric_columns, scaler)

    # One hot encoding
    df = process_one_hot_encoder(df, categorical_columns, one_hot_encoder)

    return df


def process_scaler(df, columns, scaler):
    data_to_scale = df[columns]

    scaled_array = scaler.transform(data_to_scale)

    scaled_df = pd.DataFrame(scaled_array, columns=columns, index=df.index)

    df = df.drop(columns, axis=1)
    df = pd.concat([df, scaled_df], axis=1)

    return df


def process_one_hot_encoder(df, columns, encoder):
    data_to_encode = df[columns]

    encoded_array = encoder.transform(data_to_encode)
    encoded_columns = encoder.get_feature_names_out(columns)

    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)

    df = df.drop(columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)

    return df


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame(item.model_dump(), index=[0])
    df_processed = process_df(df)

    return model.predict(df_processed)


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)) -> Response:
    contents = await file.read()

    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    df_processed = process_df(df)

    df['selling_price'] = model.predict(df_processed)

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return Response(content=output.getvalue(), media_type="text/csv")
