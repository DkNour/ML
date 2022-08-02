import pickle
import dill
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
explainer = joblib.load("C:\\Users\\nourd\\PycharmProjects\\pythonProject5\\shap39_treeExplainer.bz2")

#Load second feature selection model: LIME
with open("C:\\Users\\nourd\\PycharmProjects\\pythonProject5\\lime_explainer.bz2", 'rb') as f:
   lime_explainer = dill.load(f)


#Function to convert input list/array to dataframe
def rows_to_df(rows):
   df = pd.DataFrame(rows, columns=['age', 'gender', 'nhiss', 'mrs', 'systolic', 'distolic', 'glucose', 'paralysis', 'smoking', 'bmi', 'cholestrol', 'tos'])
   # set category columns first to short numeric to save memory etc, then convert to categorical for catboost
   for col in ['gender', 'smoking']:
      df[col] = df[col].astype('int8')
      df[col] = df[col].astype('category')
   # and finally convert all non-categoricals to their original type. since we had to create a fresh dataframe this is needed
   for col in ['age', 'gender', 'nhiss', 'mrs', 'systolic', 'distolic', 'glucose', 'paralysis', 'smoking', 'bmi', 'cholestrol', 'tos']:
      if col not in ['gender', 'smoking']:
         df[col] = df[col].astype(np.int64)
   return df


# the function to pass to LIME for running catboost on the input data
def c_run_pred(x):
   p = catboost_model.predict_proba(rows_to_df(x))
   return p

#c_predict_fn: Classifier prediction probability function, which takes a numpy array and outputs prediction probabilities
c_predict_fn = lambda x: c_run_pred(x)

def get_shap_explanation_scores_df(patient):

    """
    Input: numpy array of patient input data
    Output: dataframe of the 12 features with their respective importance score calculated by SHAP (sorted)
    """

    label = get_risk_level(patient)

    df = pd.DataFrame(np.array(patient)[np.newaxis], columns=['age', 'gender', 'nhiss', 'mrs', 'systolic', 'distolic', 'glucose', 'paralysis', 'smoking', 'bmi', 'cholestrol', 'tos'])
    shap_values = explainer.shap_values(df)
    sdf_train = pd.DataFrame({
        'feature_value':  df.values.flatten(),
        'shap_values': shap_values[label].flatten()
    })
    sdf_train.index =df.columns.to_list()
    sdf_train = sdf_train.sort_values('shap_values',ascending=False)
    sdf_train['Feature'] = sdf_train.index

    return(sdf_train)

def get_lime_explanation_scores_df(patient):
    """
    Input: numpy array of patient input data
    Output: dataframe of the 12 features with their respective importance score calculated by LIME (sorted)
    """

    label = get_risk_level(patient)

    class_names=["0", "1", "2","3"]
    exp = lime_explainer.explain_instance(patient, c_predict_fn, num_features=12, top_labels=3)
    features = []
    for i in range(12):
        list_exp = exp.as_list(label=label)[i][0].split()
        for k in list_exp:
            if k.isalpha():
                features.append(k)
    df = pd.DataFrame.from_records(exp.as_list(label=label), columns =['Feature', 'Weight'])
    df.index =features
    df['Feature'] =features
    df = df.sort_values('Weight',ascending=False)

    return(df)


def plot_SHAP_selection(patient_data):
    colors = {'nhiss': 'orange', 'distolic': 'r', 'systolic': 'c', 'cholestrol': 'y', 'gender': 'r', 'age': 'k', 'paralysis': 'b', 'smoking': 'y', 'mrs': 'coral', 'bmi': 'g', 'tos': 'pink', 'glucose': 'orchid'}
    df = get_shap_explanation_scores_df(patient_data)
    df['shap_values'][:5].plot(kind="bar", color=[colors[i] for i in df['Feature']], title="SHAP Explanation")
    plt.show()

def plot_LIME_selection(patient_data):
    colors = {'nhiss': 'orange', 'distolic': 'r', 'systolic': 'c', 'cholestrol': 'y', 'gender': 'r', 'age': 'k', 'paralysis': 'b', 'smoking': 'y', 'mrs': 'coral', 'bmi': 'g', 'tos': 'pink', 'glucose': 'orchid'}
    df = get_lime_explanation_scores_df(patient_data)
    df['Weight'][:5].plot(kind="bar", color=[colors[i] for i in df['Feature']], title="LIME Explanation")
    plt.show()

def get_risk_level(patient_data):
    """
    Input: numpy array of patient data
    Output: risk level from 0-3
    """

    prediction = catboost_model.predict_proba(patient_data)
    label = np.argmax(prediction, axis=0)

    return(label)

test_input = np.array([ 90,   1,   30,  0, 121,  83,  94,   3,   0,  22, 207, 3])

#plot_LIME_selection(test_input)
print(get_lime_explanation_scores_df(test_input))