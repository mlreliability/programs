#Plot MultiMacroROC curves
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV

import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv('CArepo1a.csv', encoding= 'unicode_escape')
df.dropna(inplace=True)
X = df.drop(['Target'], axis=1)
y = df['Target']
X = preprocessing.StandardScaler().fit(X).transform(X)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.4,
                             random_state=0)

label_encoder=LabelEncoder()
label_encoder.fit(y)
y=label_encoder.transform(y)
classes=label_encoder.classes_

sss.get_n_splits(X, y)
scores = []

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#For DTC
DTCclf = DecisionTreeClassifier()
DTCclf.fit(X_train, y_train)
DTC_y_score =DTCclf.predict_proba(X_test)

DTC_y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

# roc curve for classes
DTC_fpr = {}
DTC_tpr = {}
DTC_thresh ={}
DTC_roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):
    DTC_fpr[i], DTC_tpr[i], DTC_thresh[i] = roc_curve(DTC_y_test_binarized[:,i], DTC_y_score[:,i])
   

all_fpr = np.unique(np.concatenate([DTC_fpr[i] for i in range(n_class)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_class):
    mean_tpr += np.interp(all_fpr, DTC_fpr[i], DTC_tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_class
DTC_fpr["macro"] = all_fpr
DTC_tpr["macro"] = mean_tpr
DTC_roc_auc["macro"] = auc(DTC_fpr["macro"], DTC_tpr["macro"])
#END of DTC


#For LR
LRclf = LogisticRegression()
LRclf.fit(X_train, y_train)
LR_y_score =LRclf.predict_proba(X_test)

LR_y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

# roc curve for classes
LR_fpr = {}
LR_tpr = {}
LR_thresh ={}
LR_roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):
    LR_fpr[i], LR_tpr[i], LR_thresh[i] = roc_curve(LR_y_test_binarized[:,i], LR_y_score[:,i])
    #print(auc(LR_fpr[i], LR_tpr[i]))

LR_all_fpr = np.unique(np.concatenate([LR_fpr[i] for i in range(n_class)]))
# Then interpolate all ROC curves at this points
LR_mean_tpr = np.zeros_like(LR_all_fpr)
for i in range(n_class):
    LR_mean_tpr += np.interp(LR_all_fpr, LR_fpr[i], LR_tpr[i])

# Finally average it and compute AUC
LR_mean_tpr /= n_class
LR_fpr["macro"] = LR_all_fpr
LR_tpr["macro"] = LR_mean_tpr
LR_roc_auc["macro"] = auc(LR_fpr["macro"], LR_tpr["macro"])
#END of LR

#For RFC
RFCclf = RandomForestClassifier()
RFCclf.fit(X_train, y_train)
RFC_y_score =RFCclf.predict_proba(X_test)

RFC_y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

# roc curve for classes
RFC_fpr = {}
RFC_tpr = {}
RFC_thresh ={}
RFC_roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):
    RFC_fpr[i], RFC_tpr[i], RFC_thresh[i] = roc_curve(RFC_y_test_binarized[:,i], RFC_y_score[:,i])
    #print(auc(LR_fpr[i], LR_tpr[i]))

RFC_all_fpr = np.unique(np.concatenate([RFC_fpr[i] for i in range(n_class)]))
# Then interpolate all ROC curves at this points
RFC_mean_tpr = np.zeros_like(RFC_all_fpr)
for i in range(n_class):
    RFC_mean_tpr += np.interp(RFC_all_fpr, RFC_fpr[i], RFC_tpr[i])

# Finally average it and compute AUC
RFC_mean_tpr /= n_class
RFC_fpr["macro"] = RFC_all_fpr
RFC_tpr["macro"] = RFC_mean_tpr
RFC_roc_auc["macro"] = auc(RFC_fpr["macro"], RFC_tpr["macro"])
#END of RFC


#For SGDC
GDclf = SGDClassifier(loss="log_loss")
GDclf.fit(X_train, y_train)
GD_y_score =GDclf.predict_proba(X_test)

GD_y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

# roc curve for classes
GD_fpr = {}
GD_tpr = {}
GD_thresh ={}
GD_roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):
    GD_fpr[i], GD_tpr[i], GD_thresh[i] = roc_curve(GD_y_test_binarized[:,i], GD_y_score[:,i])
    

GD_all_fpr = np.unique(np.concatenate([GD_fpr[i] for i in range(n_class)]))
# Then interpolate all ROC curves at this points
GD_mean_tpr = np.zeros_like(GD_all_fpr)
for i in range(n_class):
    GD_mean_tpr += np.interp(GD_all_fpr, GD_fpr[i], GD_tpr[i])

# Finally average it and compute AUC
GD_mean_tpr /= n_class
GD_fpr["macro"] = GD_all_fpr
GD_tpr["macro"] = GD_mean_tpr
GD_roc_auc["macro"] = auc(GD_fpr["macro"], GD_tpr["macro"])
#END of SGDC


#For GBC
GBCclf = GradientBoostingClassifier()
GBCclf.fit(X_train, y_train)
GBC_y_score =GBCclf.predict_proba(X_test)

GBC_y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

# roc curve for classes
GBC_fpr = {}
GBC_tpr = {}
GBC_thresh ={}
GBC_roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):
    GBC_fpr[i], GBC_tpr[i], GBC_thresh[i] = roc_curve(GBC_y_test_binarized[:,i], GBC_y_score[:,i])

GBC_all_fpr = np.unique(np.concatenate([GBC_fpr[i] for i in range(n_class)]))
# Then interpolate all ROC curves at this points
GBC_mean_tpr = np.zeros_like(GBC_all_fpr)
for i in range(n_class):
    GBC_mean_tpr += np.interp(GBC_all_fpr, GBC_fpr[i], GBC_tpr[i])

# Finally average it and compute AUC
GBC_mean_tpr /= n_class
GBC_fpr["macro"] = GBC_all_fpr
GBC_tpr["macro"] = GBC_mean_tpr
GBC_roc_auc["macro"] = auc(GBC_fpr["macro"], GBC_tpr["macro"])
#END of GBC


#For LRCV
LRCVclf = LogisticRegressionCV(max_iter=10000)
LRCVclf.fit(X_train, y_train)
LRCV_y_score =LRCVclf.predict_proba(X_test)

LRCV_y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

# roc curve for classes
LRCV_fpr = {}
LRCV_tpr = {}
LRCV_thresh ={}
LRCV_roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):
    LRCV_fpr[i], LRCV_tpr[i], LRCV_thresh[i] = roc_curve(LRCV_y_test_binarized[:,i], LRCV_y_score[:,i])

LRCV_all_fpr = np.unique(np.concatenate([LRCV_fpr[i] for i in range(n_class)]))
# Then interpolate all ROC curves at this points
LRCV_mean_tpr = np.zeros_like(LRCV_all_fpr)
for i in range(n_class):
    LRCV_mean_tpr += np.interp(LRCV_all_fpr, LRCV_fpr[i], LRCV_tpr[i])

# Finally average it and compute AUC
LRCV_mean_tpr /= n_class
LRCV_fpr["macro"] = LRCV_all_fpr
LRCV_tpr["macro"] = LRCV_mean_tpr
LRCV_roc_auc["macro"] = auc(LRCV_fpr["macro"], LRCV_tpr["macro"])
#END of LRCV

#plt.plot(DTC_fpr["macro"],DTC_tpr["macro"], label="Decision Tree Classifier  (area = {0:0.4f})".format(DTC_roc_auc["macro"]),color="navy", linestyle="solid", linewidth=3,)
plt.plot(DTC_fpr["macro"],DTC_tpr["macro"], label="Decision Tree Classifier  ({0:0.4f})".format(DTC_roc_auc["macro"]),color="navy", linestyle="solid", linewidth=3,)
plt.plot(LR_fpr["macro"],LR_tpr["macro"], label="Logistic Regression  ({0:0.4f})".format(LR_roc_auc["macro"]),color="blue", linestyle="dashed", linewidth=3,)
plt.plot(RFC_fpr["macro"],RFC_tpr["macro"], label="Random Forest Classifier  ({0:0.4f})".format(RFC_roc_auc["macro"]),color="forestgreen", linestyle="dashdot", linewidth=3,)
plt.plot(GD_fpr["macro"],GD_tpr["macro"], label="Stochastic Gradient Descent Classifier  ({0:0.4f})".format(GD_roc_auc["macro"]),color="cyan", linestyle="dotted", linewidth=3,)
plt.plot(GBC_fpr["macro"],GBC_tpr["macro"], label="GradientBoostingClassifier  ({0:0.4f})".format(GBC_roc_auc["macro"]),color="teal", linestyle="solid", linewidth=3,)
plt.plot(LRCV_fpr["macro"],LRCV_tpr["macro"], label="Logistic RegressionCV  ({0:0.4f})".format(LRCV_roc_auc["macro"]),color="dodgerblue", linestyle="dotted", linewidth=3,)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#plt.title("Without feature optimization ")
plt.legend(loc="lower right")
plt.show()