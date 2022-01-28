# Import packages
import time
start_time = time.time()
import pandas as pd, os, numpy as np, csv, sys
from sklearn import metrics, model_selection
#from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
#import matplotlib.pyplot as plt

# Read feature list in a dataframe
FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\..\\features\\features.csv"
df = pd.read_csv(FEATURE_LIST_CSV_FILE_PATH)
data = df

# Logistic Regression Model

def LR(data):
    #How to change your accuracy for matching: Change the C value below between 1e8 and  1e-8
    logreg = LogisticRegression(C=1e-6, multi_class='ovr', penalty='l2', random_state=0)
    X = data.drop(['label'],axis=1) # all features
    Y = data['label'] #Label or class, ground truth
    predict = model_selection.cross_val_predict(logreg, X, Y, cv=10)
    #print(metrics.classification_report(data['label'], predict))
    #accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
# =============================================================================
#     acc = []
#     acc.append(metrics.accuracy_score(Y, predict))
#     acc = (float(sum(acc) / len(acc)))
# =============================================================================
    acc = metrics.accuracy_score(Y, predict)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    F1 = metrics.f1_score(Y, predict, zero_division=0,average = 'macro')
    P = metrics.precision_score(Y, predict, zero_division=0,average = 'macro')
    R = metrics.recall_score(Y, predict, zero_division=0,average = 'macro')
    return acc * 100, F1 * 100, P * 100, R * 100

# SVM Model

def SVM(data):
    #https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
    SVM = SVC(C=0.1, kernel='linear')
    X = data.drop(['label'],axis=1) # all features
    Y = data['label'] #Label or class, ground truth
    predict = model_selection.cross_val_predict(SVM, X, Y, cv=10)
    #print(metrics.classification_report(data['label'], predict))
    #accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
# =============================================================================
#     acc = []
#     acc.append(metrics.accuracy_score(Y, predict))
#     acc = (float(sum(acc) / len(acc)))
# =============================================================================
    acc = metrics.accuracy_score(Y, predict)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    F1 = metrics.f1_score(Y, predict, zero_division=0,average = 'macro')
    P = metrics.precision_score(Y, predict, zero_division=0,average = 'macro')
    R = metrics.recall_score(Y, predict, zero_division=0,average = 'macro')
    return acc * 100, F1 * 100, P * 100, R * 100

# Decision Tree model   
   
def DT(data):
    decision_classifier = DecisionTreeClassifier()
    X = data.drop(['label'],axis=1) # all features
    Y = data['label'] #Label or class, ground truth
    predict = model_selection.cross_val_predict(decision_classifier, X, Y, cv=10)
    #print(metrics.classification_report(data['label'], predict))
    #accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
# =============================================================================
#     acc = []
#     acc.append(metrics.accuracy_score(Y, predict))
#     acc = (float(sum(acc) / len(acc)))
#     Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
# =============================================================================
    acc = metrics.accuracy_score(Y, predict)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    F1 = metrics.f1_score(Y, predict, zero_division=0,average = 'macro')
    P = metrics.precision_score(Y, predict, zero_division=0,average = 'macro')
    R = metrics.recall_score(Y, predict, zero_division=0,average = 'macro')
    return acc * 100, F1 * 100, P * 100, R * 100

# Naive Bayes Model
   
def NB(data):
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    NB_classifier = MultinomialNB()
    X = data.drop(['label'],axis=1) # all features
    Y = data['label'] #Label or class, ground truth
    predict = model_selection.cross_val_predict(NB_classifier, X, Y, cv=10)
    '''print(f'{data.head()}')
    print(metrics.classification_report(data['label'], predict))
    print(f'Accuracy: {metrics.accuracy_score(Y, predict)}')
    #accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    '''
# =============================================================================
#     acc = []
#     acc.append(metrics.accuracy_score(Y, predict))
#     acc = (float(sum(acc) / len(acc)))
# =============================================================================
    acc = metrics.accuracy_score(Y, predict)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    F1 = metrics.f1_score(Y, predict, zero_division=0,average='macro')
    P = metrics.precision_score(Y, predict, zero_division=0,average='macro')
    R = metrics.recall_score(Y, predict, zero_division=0,average='macro')
    #res = metrics.classification_report(Y, predict)
    return acc * 100, F1 * 100, P * 100, R * 100


# Random Forest Model
   
def RF(data):
    RF_classifier = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    X = data.drop(['label'],axis=1) # all features
    Y = data['label'] #Label or class, ground truth
    predict = model_selection.cross_val_predict(RF_classifier, X, Y, cv=10)
    predict = predict.round()
    #print(metrics.classification_report(data['label'], predict))
    #accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
# =============================================================================
#     acc = []
#     acc.append(metrics.accuracy_score(Y, predict))
#     acc = (float(sum(acc) / len(acc)))
# =============================================================================
    acc = metrics.accuracy_score(Y, predict)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    F1 = metrics.f1_score(Y, predict, zero_division=0,average = 'macro')
    P = metrics.precision_score(Y, predict, zero_division=0,average = 'macro')
    R = metrics.recall_score(Y, predict, zero_division=0,average = 'macro')
    return acc * 100, F1 * 100, P * 100, R * 100


features = [
    'EconomicsUniqueWordCount',
    'EntertainmentUniqueWordCount',
    'InternationalUniqueWordCount',
    'ScienceAndTechnologyUniqueWordCount',
    'SportsUniqueWordCount',
    'EconomicsRelatedWordCount',
    'EntertainmentRelatedWordCount',
    'ScienceAndTechnologyRelatedWordCount',
    'SportsRelatedWordCount'
    ]
feature_category = {
        "UniqueWord":
                [
                'EconomicsUniqueWordCount',
                'EntertainmentUniqueWordCount',
                'InternationalUniqueWordCount',
                'ScienceAndTechnologyUniqueWordCount',
                'SportsUniqueWordCount'
                ], 
        "RelatedWord":
                [
                'EconomicsRelatedWordCount',
                'EntertainmentRelatedWordCount',
                'ScienceAndTechnologyRelatedWordCount',
                'SportsRelatedWordCount'
                ], 
                }

#create result for individual algorithm and results creation may take several hours depending on algorithms
ML_Algorithms = {"DT":"DT","LR":"LR" ,"NB":"NB","SVM":"SVM", "RF":"RF"}
#change index in ML_Algorithms["DT"] to get result for another algorithm
ML_Algorithm = ML_Algorithms["LR"] 
print ("Model: " + ML_Algorithm)
#create result according to individual feature
FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\results500_u100\\"+ ML_Algorithm +"_6699_u100_Feature_Wise_Result.csv"
headers = ["Feature", "P", "R", "F1", "Acc"]
with open(FEATURE_LIST_CSV_FILE_PATH, "w", newline='') as header:
    header = csv.writer(header)
    header.writerow(headers)
with open(FEATURE_LIST_CSV_FILE_PATH, "a", newline='') as result_csv:
    writer = csv.writer(result_csv)
    for feature in features:
        tiny_data = data[[feature, 'label']]
        Acc, F1, P, R = eval(ML_Algorithm + "(tiny_data)") #LR(tiny_data) string to function call
        writer.writerow([feature, "%.2f"%P, "%.2f"%R, "%.2f"%F1, "%.2f"%Acc])
#create result according to category
FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\results500_u100\\"+ ML_Algorithm +"_6699_u100_Category_Wise_Result.csv"
with open(FEATURE_LIST_CSV_FILE_PATH, "w", newline='') as header:
    header = csv.writer(header)
    header.writerow(headers)
for (key, value) in feature_category.items():
    with open(FEATURE_LIST_CSV_FILE_PATH, "a", newline='') as result_csv:
        writer = csv.writer(result_csv)
        #add label field at the end of category features
        value.append("label")
        tiny_data = data[value]
        #eval can execute string as python code
        Acc, F1, P, R = eval(ML_Algorithm + "(tiny_data)")
        writer.writerow([key, "%.2f"%P, "%.2f"%R, "%.2f"%F1, "%.2f"%Acc])

#create result full
FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\results500_u100\\"+ ML_Algorithm +"_6699_u100_Full_Result.csv"
with open(FEATURE_LIST_CSV_FILE_PATH, "a", newline='') as header:
    header = csv.writer(header)
    header.writerow(headers)
with open(FEATURE_LIST_CSV_FILE_PATH, "a", newline='') as result_csv:
    writer = csv.writer(result_csv)
    #tiny_data = data[[feature, 'label']]
    Acc, F1, P, R = eval(ML_Algorithm + "(data)") #LR(tiny_data) string to function call
    writer.writerow([f'{ML_Algorithm}_Full Result', "%.2f"%P, "%.2f"%R, "%.2f"%F1, "%.2f"%Acc])


print("Result has been created successfully.")

#calculate execution time
end_time = time.time() - start_time
total_minutes = int(end_time)/60
hours = total_minutes/60
minutes = total_minutes%60 
seconds = int(end_time)%60
print("--- %d Hours %d Minutes %d Seconds ---" % (hours, minutes, seconds))