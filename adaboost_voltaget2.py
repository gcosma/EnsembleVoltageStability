__author__ = 
'''
Reference:
Multi-class AdaBoosted Decision Trees:
http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
#from multi_AdaBoost import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split


#imgPanel = pd.read_csv("C:/Users/Lenovo/CNN_KERAS_MNIST/voltage stability/gcv2.csv")
#imgPanel = pd.read_csv("gcv2.csv")

#imgPanel = pd.read_csv("C:/Users/sony/Documents/My document/python/adaboost/voltage_stability/gcv2.csv")
imgPanel = pd.read_csv("C:/Users/CMP3TAHERA/OneDrive - Nottingham Trent University/python/adaboost/voltage_stability/gcv2.csv")

Y= imgPanel['Bus voltage group'].as_matrix()
del imgPanel['Bus voltage group']
X = imgPanel.as_matrix()
#%%
#X= (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) + 0.001)
X= (X - np.min(X, axis = 0))
X = X/ (np.max(X, axis = 0) + 0.001)

# STEP 1: split X and y into training and testing sets
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=4)
print("Training and Testing data shapes:")
print("trainX.shape: {}".format(trainX.shape))
print("trainY.shape: {}".format(trainY.shape))
print("testX.shape: {}".format(testX.shape))
print("testY.shape: {}".format(testY.shape))
#%%
seed = 100
np.random.seed(seed)


#df=pd.DataFrame({'a':y_train})
#N_re=[200, 500]
#

# 
X_train = trainX
X_test = testX
y_train = trainY
y_test = testY  

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=10),
    n_estimators=600,
    learning_rate=1)


#bdt_discrete = AdaBoostClassifier(
#    DecisionTreeClassifier(max_depth=2),
#    n_estimators=600,
#    learning_rate=1.5,
#    algorithm="SAMME")


bdt_real.fit(X_train, y_train)
#bdt_discrete.fit(X_train, y_train)



#n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

from multi_AdaBoost import AdaBoostClassifier as Ada

bdt_real_test = Ada(
    base_estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=600,
    learning_rate=1)
bdt_real_test.fit(X_train, y_train)


real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
test_real_errors=bdt_real_test.estimator_errors_[:]
#test_discrete_errors=bdt_discrete_test.estimator_errors_[:]

plt.figure(figsize=(15, 5))
plt.subplot(221)
#plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
#         "b", label='SAMME', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
#plt.ylim((.2,
#         max(real_estimator_errors.max(),
#             discrete_estimator_errors.max()) * 1.2))
#plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(222)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='SAMME.R', alpha=.5,color='r')
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')


plt.subplot(224)
plt.plot(range(1, n_trees_real + 1), test_real_errors,
         "r", label='test_real', alpha=.5, color='b')

plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
#plt.ylim((.2,
#         max(real_estimator_errors.max(),
#             discrete_estimator_errors.max()) * 1.2))
#plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(223)
#plt.plot(range(1, n_trees_real + 1), test_discrete_errors,
#         "r", label='test_discrete', alpha=.5)

plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
#plt.ylim((.2,


from sklearn.metrics import accuracy_score
y_predict_real = bdt_real.predict(X_test)
print(accuracy_score(y_predict_real,y_test))
print(accuracy_score(bdt_real_test.predict(X_test),y_test))
#print(accuracy_score(bdt_discrete.predict(X_test),y_test))
#print(accuracy_score(bdt_discrete_test.predict(X_test),y_test))

#%%# confusion matrix for testing data
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

target_names = [ 'First', 'Second']

print (classification_report(y_test, y_predict_real,target_names=target_names))

#%% plot confusion matrix
confusion_matrix = confusion_matrix(y_test, y_predict_real)
#target_names1 = ['1', '2', '3']
target_names1 = target_names

plt.figure(figsize=(15, 5))
plt.imshow(confusion_matrix, interpolation='nearest')
#plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names1, rotation=45)
thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "orange")
plt.tight_layout()
#plt.title('Confusion Matrix', fontsize='12')