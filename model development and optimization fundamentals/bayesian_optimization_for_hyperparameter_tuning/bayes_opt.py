import optuna
import optuna.logging

from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from optuna.samplers import TPESampler
from time import time

optuna.logging.disable_default_handler()

digits = datasets.load_digits()

n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

rfc = RandomForestClassifier(random_state=42)

hyperparam_grid = {
    'n_estimators': [50,100,150,200],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [5, 6, 7]
    }


def get_best_results(param_object, total_time):

    iterations = len(param_object.cv_results_['params'])
    best_score = param_object.best_score_
    best_index = param_object.best_index_+1
    best_params = param_object.best_params_

    print(f"---{param_object.__class__.__name__}---")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Number of iterations: {iterations}")
    print(f"Best trial index: {best_index}")
    print(f"Best score: {best_score}")
    print(f"Best hyperparameters: {best_params}")

grid_search = GridSearchCV(estimator=rfc,
                  param_grid=hyperparam_grid,
                  scoring='f1_micro',
                  n_jobs=-1,
                  verbose=0)

start = time()
grid_search.fit(X, y)
total_time = time() - start

get_best_results(grid_search, total_time)

random_search = RandomizedSearchCV(estimator=rfc,
                  param_distributions=hyperparam_grid,
                  scoring='f1_micro',
                  n_jobs=-1,
                  verbose=0,
                  n_iter=360)

# perform hyperparamter tuning
start = time()
random_search.fit(X, y)
total_time = time() - start

# store result in a data frame
get_best_results(random_search, total_time)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', low=50,high= 100,step=50)
    criterion=trial.suggest_categorical('criterion', ['gini', 'entropy'])
    min_samples_split=trial.suggest_int('min_samples_split', low=2,high=4,step=1)
    min_samples_leaf=trial.suggest_int('min_samples_leaf', low=1,high=5,step=1)
    max_depth=trial.suggest_int('max_depth',low=5,high=7,step=1)
    max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    rfc = RandomForestClassifier(n_estimators=n_estimators,
                                 criterion=criterion,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 max_depth=max_depth,
                                 max_features=max_features)

    return cross_val_score(estimator=rfc,X=X,y=y,scoring='f1_micro').mean()

study=optuna.create_study(
    sampler=TPESampler(),
    direction='maximize'
)
"""Study object:
- Study object in Optuna. It is the core object that manages the optimization process.
- sampler: This implements the algorithm for selecting the subsequent hyperparameter values. This is analogous to drawing a hyperparameter from the “good” probability distribution we discussed earlier.
- direction: As the name suggests, this specifies the direction of optimization. We set minimize for minimization and maximize for maximization. As our objective function returns “f1-score” and we want to maximize it, we specify direction='maximize'.
"""

start = time()
study.optimize(objective, n_trials=100)
total_time = time() - start


print(f"---Bayesian Optimization---")
print(f"Total time: {total_time:.2f} seconds")
print(f"Number of iterations: {100}")
print(f"Best trial index: {study.best_trial.number}")
print(f"Best score: {study.best_trial.value}")
print(f"Best hyperparameters: {study.best_params}")