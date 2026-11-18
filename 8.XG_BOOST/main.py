# XG-Boost - scikit-learn interface + hand-written gradient boosting comparison

import numpy as np

#-------------------Generate data----------------------------
np.random.seed(42)
X = np.linspace(0,10,60).reshape(-1,1)
y = np.sin(X.ravel()) + 0.4 * np.sin(2 * X.ravel()) + np.random.randn(60) * 0.35

#---------Option 1--Hand written gradient boosting (rwgression  + residual loop)------------
from sklearn.tree import DecisionTreeRegressor

N_TREES , LR, MAX_DEPTH = 40, 0.3, 3

mean_y = y.mean()
preds = np.full(len(y) , mean_y)
trees = []

print("=======Hand writte gradient boosting===========")
for t in range(1, N_TREES + 1):
    residuals = y - preds
    # negative gradient mse loss
    tree = DecisionTreeRegressor(max_depth= MAX_DEPTH)
    tree.fit(X, residuals)
    preds += LR * tree.predict(X)
    trees.append(tree) 
    if t % 10 == 0 :
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        print(f" T = {t:3d} , RMSE = {rmse:.3f} ")


#--------------------Option 2----XG boost library (real xgboost)--------------------
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    X_train, X_val,y_train,y_val = train_test_split(X,y,test_size= 0.3, random_state= 0) 
    model = xgb.XGBRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        max_depth = 3,
        subsample = 0.8,
        colsample_bytree = 0.8,
        reg_lambda = 1.0,
        early_stopping_rounds = 10,
        eval_metric = 'rmse',
        verbosity = 0

    )
    model.fit(X_train,y_train, eval_set= [(X_val,y_val)], verbose = False)
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val)**2))
    print("\n=======XG Boost library==========")
    print(f" Best round: {model.best_iteration}")
    print(f" Val rmse: {val_rmse:.3f}")
    print(f" Feature importances: {model.feature_importances_} ")
except ImportError:
    print("\nxgboost not installed - run: pip install xgboost")    



#-----------Option 3--SK learn gradientboostingregressor (equivalent implementation)----------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42,
)
gb.fit(X,y)
rmse_gb = np.sqrt(mean_squared_error(y, gb.predict(X)))
print(f"\n=====Sk learn gradient boosting====")
print(f" Train RMSE : {rmse_gb:.4f}")
print(f"Number of trees: {gb.n_estimators_}")














