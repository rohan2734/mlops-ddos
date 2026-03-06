from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy.stats as st


X,y = load_breast_cancer(return_X_y=True)

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#define base model
base_model = RandomForestClassifier(random_state=42,n_jobs=-1)
# randomized search space
param_distributions = {
    "n_estimators" : st.randint(100,600),
    "max_depth": st.randint(3,21),
    "max_features": st.uniform(0.3,0.7),
    "min_samples_split" : st.randint(2,11),
    "min_samples_leaf": st.randint(1,5),
    "bootstrap" : [True,False],
    "class_weight": [None,"balanced"],
}

# cross validation config
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# performing randomized search
rnd_search=RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=40,scoring="roc_auc",
    cv=cv,random_state=42,n_jobs=-1,
    verbose=0,refit=True,
    return_train_score=True)

# fit on train
rnd_search.fit(X_train,y_train)

best_model = rnd_sesarch.best_estimator_

def optimal_threshold_by_youden(y_true,y_score):
    from sklearn.metrics import roc_auc_score
    fpr,tpr,thr= roc_curve(y_true,y_score)
    return thr[np.argmax(tpr-fpr)]

oof_proba = cross_val_predict(best_model,X_train,y_train,cv=cv,method="predict_proba",n_jobs=-1)[:,1]

opt_thresh = optimal_threshold_by_youden(y_train, oof_proba)
test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= opt_thresh).astype(int)

test_auc = roc_auc_score(y_test, test_proba)
test_acc = accuracy_score(y_test, test_pred)

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))
