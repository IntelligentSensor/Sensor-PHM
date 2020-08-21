#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import _pickle as cPickle
import scipy.optimize as opt
from datetime import datetime
from scipy.stats import uniform, randint
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, ShuffleSplit, learning_curve

from sklearn.metrics import auc, accuracy_score,roc_auc_score
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

path = '/Users/tung/Python/WorkProject/PHMresearch/WDCNN&LR_FaultDiagnosis/'

feature_train_scaled = cPickle.load( open(path+'feature_train_scaled.pkl','rb') )
feature_valid_scaled = cPickle.load( open(path+'feature_valid_scaled.pkl','rb') )
feature_test_scaled = cPickle.load( open(path+'feature_test_scaled.pkl','rb') )

y_train_decode = cPickle.load( open(path+'y_train_decode.pkl','rb') )
y_valid_decode = cPickle.load( open(path+'y_valid_decode.pkl','rb') )
y_test_decode = cPickle.load( open(path+'y_test_decode.pkl','rb') )

def encode(data):
    data = np.array(data).reshape([-1, 1])
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    Encoder = OneHotEncoder()
    Encoder.fit(data)
    encoded = Encoder.transform(data).toarray()
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

y_test = encode(y_test_decode)


def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))


'lasso lr'

#parameters set
lasso_model = LogisticRegression(C=0.7, penalty='l1', tol=1e-6, solver = 'liblinear',
                                 class_weight = 'balanced',multi_class= 'ovr', max_iter=1000) #不可导

print(lasso_model)

epoch = 5
scores = []

# Train model
start = datetime.now()

for i in range(epoch):
    lasso_model.fit(feature_train_scaled, y_train_decode)
    #     y_score = lasso_model.predict_proba(x_test)
    #     scores.append(roc_auc_score(y_test, y_score, average='micro'))
    scores.append(lasso_model.score(feature_test_scaled, y_test_decode)) #准确率


print("This took ", datetime.now() - start)
print(u'The accuracy of the model is: ')
display_scores(scores)  #准确率

coef_ = pd.DataFrame(lasso_model.coef_, columns = ['maximum', 'minimum', 'mean', 'var', 'std', 'absmean', 'rms',
'vi', 'skew', 'kurt', 'ptp', 'par', 'form', 'impulse', 'margin', 'spectral_kurt', 'spectral_skw','spectral_pow',
'wave_L2', 'frechet_form1', 'frechet_form2', 'frechet_form3', 'frechet_form4', 'frechet_form5', 'frechet_form6',
        'frechet_form7', 'frechet_form8' ])
coef_ #特征权重

#evaluation
prediction_train = lasso_model.predict(feature_train_scaled)
cm_train = confusion_matrix(y_train_decode, prediction_train)
prediction_test = lasso_model.predict(feature_test_scaled)
cm_test = confusion_matrix(y_test_decode, prediction_test)

print("Confusion matrix for training dataset is \n%s\n for testing dataset is \n%s.\n"
      % (cm_train, cm_test))

target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test_decode, prediction_test, target_names=target_names))

'rige lr'
#parameters set
ridge_model = LogisticRegression(
                                 C=0.8,                      #正则化系数λ的倒数
                                 penalty='l2',
                                 dual = True,                #用在求解线性多核(liblinear)的L2惩罚项上。样本数量>样本特征的时False
                                 tol=1e-6,                   #绝对误差限，停止求解的标准
                                 class_weight = 'balanced',
                                 solver = 'liblinear',       #可导 lbfgs sag
                                 max_iter=5000)

print(ridge_model)

epoch = 5
scores = []

# Train model
start = datetime.now()

for i in range(epoch):
    ridge_model.fit(feature_train_scaled, y_train_decode)
    #     y_score = ridge_model.predict_proba(x_test)
    #     scores.append(roc_auc_score(y_test, y_score, average='micro'))
    scores.append(ridge_model.score(feature_test_scaled, y_test_decode)) #准确率

print("This took ", datetime.now() - start)
print(u'The accuracy of the model is: ')
display_scores(scores)  #准确率

param_test1 = {
    'C':[i/10.0 for i in range(1,11)]         # c：正则化系数λ的倒数
}
gsearch1 = GridSearchCV(estimator = ridge_model,
                        param_grid = param_test1, scoring='accuracy',n_jobs=2, cv=5)

gsearch1.fit(feature_valid_scaled, y_valid_decode)

print(gsearch1.cv_results_['mean_test_score'])
print(gsearch1.best_params_)
print("best accuracy:%f" % gsearch1.best_score_)

coef_ = pd.DataFrame(ridge_model.coef_, columns = ['maximum', 'minimum', 'mean', 'var', 'std', 'absmean', 'rms',
'vi', 'skew', 'kurt', 'ptp', 'par', 'form', 'impulse', 'margin', 'spectral_kurt', 'spectral_skw','spectral_pow',
'wave_L2', 'frechet_form1', 'frechet_form2', 'frechet_form3', 'frechet_form4', 'frechet_form5', 'frechet_form6',
                                                   'frechet_form7', 'frechet_form8' ])

coef_

#evaluation
prediction_train = ridge_model.predict(feature_train_scaled)
cm_train = confusion_matrix(y_train_decode, prediction_train)
prediction_test = ridge_model.predict(feature_test_scaled)
cm_test = confusion_matrix(y_test_decode, prediction_test)

print("Confusion matrix for training dataset is \n%s\n for testing dataset is \n%s.\n"
      % (cm_train, cm_test))

target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test_decode, prediction_test, target_names=target_names))

y_score = ridge_model.predict_proba(feature_test_scaled)

# 计算micro类型的AUC
# print('调用函数auc：', roc_auc_score(y_test, y_score, average='micro'))

fpr, tpr, thresholds = roc_curve(y_test.ravel(),y_score.ravel())
micro_auc = auc(fpr, tpr)
print ('micro_auc：', micro_auc)

'demo lr'
def demoLr(data):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # add a ones column - this makes the matrix multiplication work out easier
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    # convert to numpy arrays and initalize the parameter array theta
    X = X.values
    y = y.values.reshape(len(y),)

    theta = np.zeros(X.shape[1])

    #regularized cost
    def cost(theta, X, y):
        ''' cost fn is -l(theta) for you to minimize'''
        return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

    cost(theta, X, y)

    def regularized_cost(theta, X, y, l=1):
        #     '''you don't penalize theta_0'''
        theta_j1_to_n = theta[1:]
        regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()
        
        return cost(theta, X, y) + regularized_term

    'this is the same as the not regularized cost because we init theta as zeros...'
    regularized_cost(theta, X, y, l=1)

    #regularized gradient
    def gradient(theta, X, y):
        #     '''just 1 batch gradient'''
        return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    gradient(theta, X, y)

    def regularized_gradient(theta, X, y, l=1):
        #     '''still, leave theta_0 alone'''
        theta_j1_to_n = theta[1:]
        regularized_theta = (l / len(X)) * theta_j1_to_n
        
        # by doing this, no offset is on theta_0
        regularized_term = np.concatenate([np.array([0]), regularized_theta])
        
        return gradient(theta, X, y) + regularized_term

    regularized_gradient(theta, X, y)

    print('init cost = {}'.format(regularized_cost(theta, X, y)))

    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
    res

    #prediction
    def predict(x, theta):
        prob = sigmoid(x @ theta)
        return (prob >= 0.5).astype(int)

    final_theta = res.x
    y_pred = predict(X, final_theta)

    print(classification_report(y, y_pred))

'SVM'
def demoSVM(data):
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]

    svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)            #liner
    svc

    svc.fit(data[['spectral_pow', 'mean']], data['y'])
    print(svc.score(data[['spectral_pow', 'mean']], data['y']))

    svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=100000)       #惩罚系数
    svc2.fit(data[['spectral_pow', 'mean']], data['y'])
    print(svc2.score(data[['spectral_pow', 'mean']], data['y']))

    svc3 = svm.SVC(C=3, kernel='rbf', gamma=5000, probability=True)  #GaussianKernel
    svc3

    svc3.fit(data[['spectral_pow', 'mean']], data['y'])
    print(svc3.score(data[['spectral_pow', 'mean']], data['y']))

    ypred = svc3.predict(data[['spectral_pow', 'mean']])
    print(classification_report(data['y'], ypred))

    candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000, 5000]  #c gamma
    combination = [(C, gamma) for C in candidate for gamma in candidate]

    search = []
    for C, gamma in combination:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(data[['spectral_pow', 'mean']], data['y'])
        search.append(svc.score(data[['spectral_pow', 'mean']], data['y']))   #CV数据

    best_score = search[np.argmax(search)]
    best_param = combination[np.argmax(search)]

    print(best_score, best_param)

'RF'
rf = RandomForestClassifier(
                            n_estimators = 21,
                            max_depth=5,
                            max_features='auto',             #构建决策树最优模型时考虑的最大特征数 n的平方根
                            oob_score = True,                #选用袋外样本，其误差是测试数据集误差的无偏估计
                            min_samples_leaf = 1,            #叶子节点含有的最少样本,小于则剪枝
                            min_samples_split= 2,            #叶子节点可分的最小样本数
                            max_leaf_nodes = None,           #最大叶子节点数
                            min_impurity_decrease = 0.0,     #节点划分的最小不纯度
                            criterion = 'gini',              #表示节点的划分标准
                            min_weight_fraction_leaf=0.0     #叶子节点最小的样本权重和,小于则剪枝,较多样本的缺失值或偏差很大时尝试
                            )
print(rf)

epoch = 5
scores = []

# Train model
start = datetime.now()

for i in range(epoch):
    rf.fit(feature_train_scaled, y_train_decode)
    scores.append(rf.score(feature_test_scaled, y_test_decode)) #准确率

print("This took ", datetime.now() - start)
print(u'The accuracy of the model is: ')
display_scores(scores)      #准确率

param_test2 = {             #bagging参数
    "n_estimators":range(1,101,10)
}
gsearch2 = GridSearchCV(estimator = rf,
                        param_grid = param_test2, scoring='accuracy',n_jobs=2, cv=5)

gsearch2.fit(feature_valid_scaled, y_valid_decode)

print(gsearch2.cv_results_['mean_test_score'])
print(gsearch2.best_params_)
print("best accuracy:%f" % gsearch2.best_score_)

#evaluation
prediction_train = rf.predict(feature_train_scaled)
cm_train = confusion_matrix(y_train_decode, prediction_train)
prediction_test = rf.predict(feature_test_scaled)
cm_test = confusion_matrix(y_test_decode, prediction_test)

print("Confusion matrix for training dataset is \n%s\n for testing dataset is \n%s.\n"
      % (cm_train, cm_test))

#Precision_rate预测为正例的样本中的真正正例的比例
#Recall_rate预测为正例的真正正例占所有真正正例的比例

target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test_decode, prediction_test, target_names=target_names))

y_score = rf.predict_proba(feature_test_scaled)

# 计算micro类型的AUC
# print('调用函数auc：', roc_auc_score(y_test, y_score, average='micro'))

fpr, tpr, thresholds = roc_curve(y_test.ravel(),y_score.ravel())
micro_auc = auc(fpr, tpr)
print ('micro_auc：', micro_auc)

'Adaboost'
Ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=21, learning_rate=0.8)

print (Ada)
epoch = 5
scores = []

# Train model
start = datetime.now()

for i in range(epoch):
    Ada.fit(feature_train_scaled, y_train_decode)
    scores.append(Ada.score(feature_test_scaled, y_test_decode)) #准确率

print("This took ", datetime.now() - start)
print(u'The accuracy of the model is: ')
display_scores(scores)  #准确率

params = {
    "learning_rate": uniform(0.1, 0.9),     # default 0.1
    "n_estimators": randint(10, 120)        # default 5
#     "max_depth": randint(2, 6)            # default 3
}

search = RandomizedSearchCV(Ada, param_distributions=params, random_state=42,
                            n_iter=30, cv=5, verbose=1, n_jobs=2, return_train_score=True)

search.fit(feature_valid_scaled, y_valid_decode)

print(search.best_estimator_)
print(search.best_score_)

#evaluation
prediction_train = Ada.predict(feature_train_scaled)
cm_train = confusion_matrix(y_train_decode, prediction_train)
prediction_test = Ada.predict(feature_test_scaled)
cm_test = confusion_matrix(y_test_decode, prediction_test)

print("Confusion matrix for training dataset is \n%s\n for testing dataset is \n%s.\n"
      % (cm_train, cm_test))

#Precision_rate预测为正例的样本中的真正正例的比例
#Recall_rate预测为正例的真正正例占所有真正正例的比例

target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test_decode, prediction_test, target_names=target_names))

y_score = Ada.predict_proba(feature_test_scaled)

# 计算micro类型的AUC
# print('调用函数auc：', roc_auc_score(y_test, y_score, average='micro'))

fpr, tpr, thresholds = roc_curve(y_test.ravel(),y_score.ravel())
micro_auc = auc(fpr, tpr)
print ('micro_auc：', micro_auc)

'GBDT'
GBDT = GradientBoostingClassifier(
                                  
                                  loss = 'deviance',
                                  learning_rate = 0.04,
                                  n_estimators = 68,
                                  max_depth = 4,
                                  subsample = 0.42,
                                  
                                  max_features = None,
                                  max_leaf_nodes=None,
                                  min_samples_leaf=1,
                                  min_samples_split=2,
                                  min_impurity_decrease=0.0,
                                  min_impurity_split=None,
                                  criterion='friedman_mse',
                                  min_weight_fraction_leaf=0.0,
                                  ccp_alpha=0.0
                                  )
print(GBDT)

epoch = 5
scores = []

# Train model
start = datetime.now()

for i in range(epoch):
    GBDT.fit(feature_train_scaled, y_train_decode)
    scores.append(GBDT.score(feature_test_scaled, y_test_decode)) #准确率

print("This took ", datetime.now() - start)
print(u'The accuracy of the model is: ')
display_scores(scores)  #准确率

params = {
    "learning_rate": uniform(0.03, 0.5),   # default 0.1
    "max_depth": randint(2, 6),            # default 3
    "n_estimators": randint(10, 120),      # default 100
    "subsample": uniform(0.1, 0.8)
}

search = RandomizedSearchCV(GBDT, param_distributions=params, random_state=42,
                            n_iter=30, cv=5, verbose=1, n_jobs=2, return_train_score=True)

search.fit(feature_valid_scaled, y_valid_decode)

print(search.best_estimator_)
print(search.best_score_)

#evaluation
prediction_train = GBDT.predict(feature_train_scaled)
cm_train = confusion_matrix(y_train_decode, prediction_train)
prediction_test = GBDT.predict(feature_test_scaled)
cm_test = confusion_matrix(y_test_decode, prediction_test)

print("Confusion matrix for training dataset is \n%s\n for testing dataset is \n%s.\n"
      % (cm_train, cm_test))

target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8', 'class 9']
print(classification_report(y_test_decode, prediction_test, target_names=target_names))

y_score = GBDT.predict_proba(feature_test_scaled)

# 计算micro类型的AUC
# print('调用函数auc：', roc_auc_score(y_test, y_score, average='micro'))

fpr, tpr, thresholds = roc_curve(y_test.ravel(),y_score.ravel())
micro_auc = auc(fpr, tpr)
print ('micro_auc：', micro_auc)
