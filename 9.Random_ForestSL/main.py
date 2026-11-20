# Random Forest - scikit-learn interface + hand-written version comparison

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#-----Option 1: Hand-written random forest (bootstrap + feature subsampling + majority vote)------
from sklearn.tree import DecisionTreeClassifier

N_TREES, MAX_DEPTH = 50, 5

def bootstrap(X, y):
    idx = np.random.choice(len(X), len(X), replace= True)
    return X[idx], y[idx], idx

trees, oob_votes = [], np.zeros((len(X_train), 2), dtype=int)

for  _ in range(N_TREES):
    Xb,yb,idx = bootstrap(X_train, y_train)
    oob_idx = np.setdiff1d(np.arange(len(X_train)), idx)
    tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, max_features='sqrt', random_state=None)
    tree.fit(Xb,yb)
    trees.append(tree)
    if len(oob_idx):
        preds = tree.predict(X_train[oob_idx])
        for j, pred in zip(oob_idx, preds):
            oob_votes[j, int(pred)] += 1

# OOB accuracy (unbiased generalization estimate - no validation set needed)
oob_mask = oob_votes.sum(axis=1) > 0
oob_pred = oob_votes[oob_mask].argmax(axis=1)
oob_acc =  accuracy_score(y_train[oob_mask], oob_pred)

def forest_predict(X):
    votes = np.zeros((len(X),2), dtype=int)
    for t in trees:
        for  i ,p in enumerate(t.predict(X)):
            votes[i , int(p)] += 1
    return votes.argmax(axis = 1)

print("========Handwritten Random Forest============")
print(f" OOB accuracy: {oob_acc:.3f}")
print(f" Test accuracy: {accuracy_score(y_test, forest_predict(X_test)):.3f} ")



#-------------------OPtion 2 skleaarn interface implementation of RFC==========
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
)
rf.fit(X_train, y_train)

print("==========sklearn RandomForestClassifier===============")
print(f" OOB accuracy: {rf.oob_score_:.3f}")
print(f" Test accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")
print(f" Feature importances: {rf.feature_importances_}")

#-----------Single decision tree baseline----------------------
from sklearn.tree import DecisionTreeClassifier

single = DecisionTreeClassifier(max_depth=5, random_state=42)
single.fit(X_train,y_train)
print("=========Single decision tree================")
print(f" Train accuracy: {accuracy_score(y_train, single.predict(X_train)):.3f}")
print(f" Test accuracy: {accuracy_score(y_test, single.predict(X_test)):.3f}")