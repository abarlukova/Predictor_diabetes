#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold


# parameters

n_splits = 5

C=10

output_file = f'model_C=10.bin'


df_init = pd.read_csv("diabetes.csv")


# ### Some checks


#print("len(df_init) = ", len(df_init))
#print(df_init.isnull().sum())
#print("Outcome unique values : ", df_init.Outcome.unique())


# ### Data preparation

# ### lowercase, and adding underscore between words


def add_underscore(string):
    l_newstr = []
    for letters in string:
        if letters.isupper():
            letters = "_"+letters
        l_newstr.append(letters)
    newstr = ''.join(l_newstr)

    if newstr.startswith('_'):
        newstr = newstr[1:]
    return newstr

df_init.rename(columns=lambda col: add_underscore(col), inplace=True)

df_init.columns = df_init.columns.str.lower()

df_init.rename(columns={"b_m_i": "bmi"}, inplace=True)

#print(df_init.columns)
#print(df_init.dtypes)


# ### Split data

# Do train/validation/test split with 60%/20%/20% distribution.
# 
# Use the train_test_split function and set the random_state parameter to 42.



df_full_train, df_test = train_test_split(df_init, test_size = 0.2, random_state = 42)


df_full_train_with_outcome = df_full_train.copy()

#print(len(df_full_train), len(df_test))




df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)




len(df_train), len(df_val), len(df_test)




#print(df_train.head())




df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)



y_train = df_train.outcome.values

y_val = df_val.outcome.values
y_test = df_test.outcome.values




del df_train["outcome"]
del df_val["outcome"]
del df_test["outcome"]




y_full_train = df_full_train.outcome.values

del df_full_train["outcome"]


# ### Logistic regression



def train(df_train, y_train, C=10):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    
    return dv, model



def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred




# Validation


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train_with_outcome):

    df_train = df_full_train_with_outcome.iloc[train_idx]
    df_val = df_full_train_with_outcome.iloc[val_idx]

    y_train = df_train.outcome.values

    y_val = df_val.outcome.values

    del df_train["outcome"]
    del df_val["outcome"]

    dv, model = train(df_train, y_train, C=10)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))





# Training the final model


dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print("auc = ", auc)

# ### Save the model





f_out = open(output_file, 'wb') 
pickle.dump((dv, model), f_out)
f_out.close()