import pickle
import dill
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from catboost import Pool

"""
requirements
numba==0.51.0
python==3.7
numpy==1.20.2
shap==0.39.0
"""

#Load risk level classifier: CatBoost
catboost_model = pickle.load(open("C:\\Users\\nourd\\PycharmProjects\\pythonProject5\\model_pkl",'rb'))

#Load first feature selection model: SHAP
explainer = joblib.load("C:\\Users\\nourd\\PycharmProjects\\pythonProject5\\explainer.bz2")

def get_shap_explanation_scores_df(patient):

    """
    Input: numpy array of patient input data
    Output: dataframe of the 12 features with their respective importance score calculated by SHAP (sorted)
    """

    label = get_risk_level(patient)

    df = pd.DataFrame(np.array(patient)[np.newaxis], columns=['age', 'gender', 'mrs', 'systolic', 'distolic', 'glucose', 'smoking', 'bmi', 'cholestrol', 'tos'])
    shap_values = explainer.shap_values(df)
    sdf_train = pd.DataFrame({
        'feature_value':  df.values.flatten(),
        'shap_values': shap_values[label].flatten()
    })
    sdf_train.index =df.columns.to_list()
    sdf_train = sdf_train.sort_values('shap_values',ascending=False)
    sdf_train['Feature'] = sdf_train.index

    return(sdf_train)

def plot_SHAP_result(patient_data):
    colors = {'distolic': 'r', 'systolic': 'c', 'cholestrol': 'y', 'gender': 'orange', 'age': 'k', 'smoking': 'y', 'mrs': 'coral', 'bmi': 'g', 'tos': 'pink', 'glucose': 'orchid'}
    df = get_shap_explanation_scores_df(patient_data)
    df['shap_values'][:].plot(kind="bar", color=[colors[i] for i in df['Feature']], title="SHAP Explanation")
    plt.show()

def get_risk_level(patient_data):
    """
    Input: numpy array of patient data
    Output: risk level from 0-3
    """

    prediction = catboost_model.predict_proba(patient_data)
    label = np.argmax(prediction, axis=0)

    return(label)

#for ploting class risk factors
def get_shap_group_explanation(explainer, X_group, y_group, x, color=False):
    d2 = pd.DataFrame(
        columns=['age', 'gender', 'mrs', 'systolic', 'distolic', 'glucose', 'smoking', 'bmi', 'cholestrol', 'tos'])
    dic = {"age": 0, "gender": 0, "mrs": 0, "systolic": 0, "distolic": 0, "glucose": 0, "smoking": 0, "bmi": 0,
           "cholestrol": 0, "tos": 0}
    resultat = {"age": 0, "gender": 0, "mrs": 0, "systolic": 0, "distolic": 0, "glucose": 0, "smoking": 0, "bmi": 0,
                "cholestrol": 0, "tos": 0}
    categorical_features = [1, 6]
    shap_values = explainer.shap_values(Pool(X_group, X_group, cat_features=categorical_features))

    for patient in range(len(X_group)):
        prediction = catboost_model.predict_proba(X_group.values[patient])
        label = np.argmax(prediction, axis=0)

        sdf_train = pd.DataFrame({
            'feature_value': X_group.iloc[[patient], :].values.flatten(),
            'shap_values': shap_values[label][[patient]].flatten()
        })
        sdf_train.index = X_group.columns.to_list()
        sdf_train['Feature'] = sdf_train.index

        aux = dict(zip(sdf_train.Feature, sdf_train.shap_values))
        dic.update(resultat)
        resultat = {**dic, **aux}

        for key, value in aux.items():
            resultat[key] = np.abs(value) + dic[key]

    resultat = dict(reversed(sorted(resultat.items(), key=lambda item: item[1])))

    keys = list(resultat.keys())
    values = list(resultat.values())

    fd = pd.DataFrame.from_dict(resultat, orient='index')
    fd['Feature'] = fd.index
    fd.rename(columns={0: 'shap_values'}, inplace=True, errors='raise')
    if color == True:
        colors = {'distolic': 'r', 'systolic': 'c', 'cholestrol': 'y', 'gender': 'b', 'age': 'k', 'smoking': 'y',
                  'mrs': 'coral', 'bmi': 'g', 'tos': 'pink', 'glucose': 'orchid'}
        fd['shap_values'][:x].plot(kind="bar", color=[colors[i] for i in fd['Feature']], figsize=(5, 3))
    else:
        fd['shap_values'][:x].plot(kind="bar", figsize=(5, 3))

    plt.show()


X_test = pd.read_csv('C:\\Users\\nourd\PycharmProjects\\pythonProject5\\x_test.csv', encoding = 'utf-8')
y_test = X_test.iloc[:,-1]
X_test = X_test.drop('risk',axis=1)


X_class0 = pd.DataFrame(columns = ['age', 'gender', 'mrs', 'systolic', 'distolic', 'glucose', 'smoking', 'bmi', 'cholestrol', 'tos'])
X_class1 = pd.DataFrame(columns = ['age', 'gender', 'mrs', 'systolic', 'distolic', 'glucose', 'smoking', 'bmi', 'cholestrol', 'tos'])
X_class2 = pd.DataFrame(columns = ['age', 'gender', 'mrs', 'systolic', 'distolic', 'glucose', 'smoking', 'bmi', 'cholestrol', 'tos'])
X_class3 = pd.DataFrame(columns = ['age', 'gender', 'mrs', 'systolic', 'distolic', 'glucose', 'smoking', 'bmi', 'cholestrol', 'tos'])

y_class0 =[]
y_class1 =[]
y_class2 =[]
y_class3 =[]

y = list(X_test.index.values)

for patient in range(len(X_test)):
  if y_test[y[patient]] == 0:
    X_class0.loc[patient] = X_test.iloc[patient].values
    y_class0.append(y_test[y[patient]])
  elif y_test[y[patient]] == 1:
    X_class1.loc[patient] = X_test.iloc[patient].values
    y_class1.append(y_test[y[patient]])
  elif y_test[y[patient]] == 2:
    X_class2.loc[patient] = X_test.iloc[patient].values
    y_class2.append(y_test[y[patient]])
  else:
    X_class3.loc[patient] = X_test.iloc[patient].values
    y_class3.append(y_test[y[patient]])

test_input = np.array([ 90,   1,   30,  0, 191,  83, 0,  22, 207, 3])

#plot_LIME_selection(test_input)
#print(plot_SHAP_result(test_input))
get_shap_group_explanation(explainer, X_class0, y_class0, 10, True)