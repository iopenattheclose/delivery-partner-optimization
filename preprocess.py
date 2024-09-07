import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dill
from imblearn.over_sampling import SMOTE
from constants import input_file_path


def fetch_data(input_file_path):
    with open(input_file_path,'rb') as f:
        df = dill.load(f)
        return df
        
def pre_process(df):
    X=df[['Region','Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']].copy()
    y=df[['class']].copy()

    #splits the data into 80% training data and 20% testing data.
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    #split into 80% training data and 20% validation data.
    X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,test_size=0.2, random_state=42)
    # 60% of the initial data as training data
    # 20% of the initial data as validation data
    # 20% of the initial data as testing data
    # print(X.shape, y.shape)
    # print(X_train.shape, y_train.shape)
    # print(X_val.shape, y_val.shape)
    # print(X_test.shape, y_test.shape)

    #smote to handle class imbalance
    smt = SMOTE()
    X_sm, y_sm = smt.fit_resample(X_train, y_train)

    # print(y_sm.value_counts())
    # print(y_sm.head())

    #standard scaling values
    st =  StandardScaler()
    X_sm = st.fit_transform(X_sm.values)
    X_val_scaled = st.transform(X_val.values)
    X_test_scaled = st.transform(X_test.values)

    y_sm = y_sm.values[:,0]
    y_val = y_val.values[:,0]
    y_test = y_test.values[:,0]

    return X_sm,X_val_scaled,X_test_scaled, y_sm,y_val,y_test

if __name__ == '__main__':
    df = fetch_data(input_file_path)
    pre_process(df)