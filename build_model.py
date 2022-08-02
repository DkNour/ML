import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder
import joblib
from catboost import Pool, CatBoostClassifier
import pickle

df_all = pd.read_csv('C:\\Users\\nourd\PycharmProjects\\pythonProject5\\Stroke_analysis1 (1).csv', encoding = 'utf-8')
label_encode_cols = ["gender"]
# Convert categorical columns to numeric encoded labels
label_encoders = {}
for col in label_encode_cols:
    le = LabelEncoder()
    label_encoders[col] = le
    df_all[col] = le.fit_transform(df_all[col])

df_all.drop(columns=['pid'], inplace=True)
df_all = df_all.drop('Unnamed: 0',axis=1)
df_all = df_all.drop('nhiss',axis=1)
df_all = df_all.drop('paralysis',axis=1)

df_all['smoking'] = (df_all['smoking'] != 0).astype(int)

from sklearn.model_selection import train_test_split

Y_col = 'risk'
X_cols = df_all.loc[:, df_all.columns != Y_col].columns

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df_all[X_cols], df_all[Y_col],test_size=0.2, random_state=42, shuffle=True)

x_test = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
x_test.to_csv ('x_test.csv', index = False, header=True)

features = [feat for feat in list(df_all)
            if feat != 'risk']
print(features)

#indices of the categorical data in our dataset : gender and smoking
cat_cols = ['gender', 'smoking']
categorical_features = [1,6]

d1={0:"female",1:"male"}
d2={0:"non smoker",1:"smoker"}

names={"gender":d1,"smoking":d2}

cat_indices = [1,6]

model = CatBoostClassifier(loss_function = 'MultiClass',iterations=1000, learning_rate=0.3, random_seed=123)
model.fit(X_train, y_train, cat_features=categorical_features, verbose=False, plot=False)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(X_test, y_test, cat_features=categorical_features))

features = [feat for feat in list(df_all) if feat != 'risk']

with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)

ex_filename = 'explainer.bz2'
joblib.dump(explainer, filename=ex_filename, compress=('bz2', 9))


























