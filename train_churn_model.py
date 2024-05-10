# Load the libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier, Pool

from sklearn.metrics import recall_score, precision_score

# Load churn prediction dataset

churn_df = pd.read_csv('Customertravel.csv')

X = churn_df.drop(columns=['Target'],axis=1)

y = churn_df['Target']

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=123)

train_data = Pool(data=X_train,label=y_train,cat_features=[1,2,4,5])
val_data = Pool(data=X_val,label=y_val,cat_features=[1,2,4,5])
test_data = Pool(data=X_test,label=y_test,cat_features=[1,2,4,5])

# Train a catboost model

model = CatBoostClassifier(n_estimators=500,
                           learning_rate=0.1,
                           depth=4,
                           loss_function='Logloss',
                           random_seed=123,
                           verbose=True)

model.fit(train_data,eval_set=val_data)

# Make predictions

y_train_pred = model.predict(train_data)
y_val_pred = model.predict(val_data)
y_test_pred = model.predict(test_data)

# Calculate precision and recall
train_precision_score = precision_score(y_train, y_train_pred)
train_recall_score = recall_score(y_train, y_train_pred)

val_precision_score = precision_score(y_val, y_val_pred)
val_recall_score = recall_score(y_val, y_val_pred)

test_precision_score = precision_score(y_test, y_test_pred)
test_recall_score = recall_score(y_test, y_test_pred)

# Print precision and recall

print(f'Train Precision: {train_precision_score}')
print(f'Val Precision: {val_precision_score}')
print(f'Test Precision: {test_precision_score}')

print(f'Train Recall: {train_recall_score}')
print(f'Val Recall: {val_recall_score}')
print(f'Test Recall: {test_recall_score}')