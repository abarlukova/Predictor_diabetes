#!/usr/bin/env python
# coding: utf-8

# In[124]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold


# In[92]:


df_init = pd.read_csv("diabetes.csv")


# ### Some checks

# In[96]:


print("len(df_init) = ", len(df_init))
print(df_init.isnull().sum())
print("Outcome unique values : ", df_init.Outcome.unique())


# ### Data preparation

# ### lowercase, and adding underscore between words

# In[97]:


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

print(df_init.columns)


# In[20]:


df_init.head().T


# In[98]:


print(df_init.dtypes)


# ### Split data

# Do train/validation/test split with 60%/20%/20% distribution.
# 
# Use the train_test_split function and set the random_state parameter to 42.

# In[131]:


df_full_train, df_test = train_test_split(df_init, test_size = 0.2, random_state = 42)


# In[12]:


len(df_full_train), len(df_test)


# In[132]:


df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)


# In[14]:


len(df_train), len(df_val), len(df_test)


# In[28]:


df_train.head()


# In[133]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[134]:


y_train = df_train.outcome.values
y_val = df_val.outcome.values
y_test = df_test.outcome.values


# In[135]:


del df_train["outcome"]
del df_val["outcome"]
del df_test["outcome"]


# In[136]:


df_full_train.head()


# In[137]:


y_full_train = df_full_train.outcome.values


# In[138]:


del df_full_train["outcome"]


# ### Logistic regression

# In[109]:


def train(df_train, y_train, C=10):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    #model = LogisticRegression(C=C, max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    
    return dv, model


# In[99]:


def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[114]:


n_splits = 5


# In[118]:


len(y_train)


# In[119]:


len(df_train)


# In[122]:


C=10


# In[127]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.outcome.values
    y_val = df_val.outcome.values

    del df_train["outcome"]
    del df_val["outcome"]

    dv, model = train(df_train, y_train, C=10)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[128]:


scores


# In[110]:


dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# ### Save the model

# In[111]:


import pickle


# In[112]:


output_file = f'model_C=10.bin'


# In[113]:


f_out = open(output_file, 'wb') 
pickle.dump((dv, model), f_out)
f_out.close()


# In[129]:


input_file = f'model_C=10.bin'


# In[130]:


with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)


# In[26]:


model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)


# In[61]:


dicts_train = df_train.to_dict(orient='records')


# In[62]:


X_train = dv.fit_transform(dicts_train)


# In[63]:


model.fit(X_train, y_train)


# In[64]:


dicts_val = df_val.to_dict(orient='records')
X_val = dv.transform(dicts_val)


# In[65]:


y_pred_val = model.predict_proba(X_val)[:, 1]


# In[38]:


#diabetes_prediction = (y_pred >= 0.5)


# In[39]:


#(y_val == diabetes_prediction.astype(int)).mean()


# In[66]:





# In[67]:


auc_pred_val = roc_auc_score(y_val, y_pred_val)
print(f'AUC for y_pred_val: {round(auc_pred_val,3)}')


# ### Using the model

# In[80]:


model_c_10 = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)


# In[68]:


dicts_full_train = df_full_train.to_dict(orient='records')


# In[69]:


X_full_train = dv.fit_transform(dicts_full_train)


# In[81]:


model_c_10.fit(X_full_train, y_full_train)


# In[139]:


dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)


# In[87]:


y_pred_test = model_c_10.predict_proba(X_test)[:, 1]


# In[88]:


auc_pred_test = roc_auc_score(y_test, y_pred_test)
print(f'AUC for y_pred_test: {round(auc_pred_test,3)}')


# In[79]:


c_arr = [0.01, 0.1, 1, 10, 100]

for c in c_arr:
    print('c=',c)
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred_val_c = model.predict_proba(X_val)[:, 1]

    auc_pred_val_c = roc_auc_score(y_val, y_pred_val_c)
    print(f'AUC for y_pred_val: {round(auc_pred_val_c,3)}')
    print('----------------')
    


# ### Transform to dict

# In[152]:


trial_participant = dicts_test[11]


# In[158]:


X_small = dv.transform([trial_participant])


# In[159]:


trial_participant


# In[154]:


model.predict_proba(pd.DataFrame(X_small)).round(3)[0,1]


# In[155]:


model.predict(pd.DataFrame(X_small))[0]


# In[156]:


y_test[10]


# ##### Random forest regressor

# In[25]:


from sklearn.ensemble import RandomForestRegressor


# In[26]:


rf = RandomForestRegressor(n_estimators=4, random_state=1, n_jobs=-1)


# In[28]:


rf.fit(df_train, y_train)


# In[31]:


y_pred_val_rf = rf.predict(df_val)


# In[32]:


auc_pred_val_rf = roc_auc_score(y_val, y_pred_val_rf)
print(f'AUC for y_pred_val_rf: {round(auc_pred_val_rf,3)}')


# In[36]:


for estim in range(2, 50, 1):
    print(f"estim = {estim}")
    rf = RandomForestRegressor(n_estimators=estim, random_state=1, n_jobs=-1)
    rf.fit(df_train, y_train)
    y_pred_val_rf = rf.predict(df_val)
    auc_pred_val_rf = roc_auc_score(y_val, y_pred_val_rf)
    print(f'AUC for y_pred_val_rf: {round(auc_pred_val_rf,3)}')
    print("------------------------")


# In[37]:


for md in range(2,10):
    print(f"max_depth = {md}")
    
    
    rmse_curr_md = []
    
    

        
    rf = RandomForestRegressor(n_estimators=10, max_depth = md, random_state=1, n_jobs=-1)
    rf.fit(df_train, y_train)

    y_pred_val_rf = rf.predict(df_val)
    auc_pred_val_rf = roc_auc_score(y_val, y_pred_val_rf)
    print(f'AUC for y_pred_val_rf: {round(auc_pred_val_rf,3)}')
    
    print("------------------------")


# ### XGBoost model

# In[54]:


#import xgboost as xgb

from xgboost import XGBClassifier


# In[55]:


model = XGBClassifier(random_state=42)
model.fit(df_train, y_train)


# In[57]:


y_pred_val_proba = model.predict_proba(df_val)[:, 1]


# In[58]:


# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_val, y_pred_val_proba)

print(f"ROC AUC Score: {roc_auc:.3f}")


# In[39]:


#dtrain = xgb.DMatrix(df_train, label=y_train, feature_names = list(df_train.columns.values))


# In[40]:


#dval = xgb.DMatrix(df_val, label=y_val, feature_names = list(df_train.columns.values))


# In[52]:


# xgb_params_1 = {
#         'eta': 0.3, 
#         'max_depth': 6,
#         'min_child_weight': 1,
        
#         'objective': 'binary:logistic',
#         'nthread': 8,
        
#         'seed': 1,
#         'verbosity': 1,
#     }


# In[42]:


#watchlist = [(dtrain, 'train'), (dval, 'val')]


# In[49]:


#from matplotlib import pyplot as plt


# In[47]:


# def parse_xgb_output(output):
#     results = []

#     for line in output.stdout.strip().split('\n'):
#         it_line, train_line, val_line = line.split('\t')

#         it = int(it_line.strip('[]'))
#         train = float(train_line.split(':')[1])
#         val = float(val_line.split(':')[1])

#         results.append((it, train, val))
    
#     columns = ['num_iter', 'train_auc', 'val_auc']
#     df_results = pd.DataFrame(results, columns=columns)
#     return df_results


# In[53]:


#%%capture output_1

#model_1 = xgb.train(xgb_params_1, dtrain, evals = watchlist, num_boost_round=100)

#s = output_1.stdout


# In[50]:


# df_score = parse_xgb_output(output_1)

# plt.plot(df_score.num_iter, df_score.train_auc, label='train')
# plt.plot(df_score.num_iter, df_score.val_auc, label='val')
# plt.legend()


# In[ ]:




