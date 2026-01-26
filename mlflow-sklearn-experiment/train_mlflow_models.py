import mlflow
from sklearn.ensemble import RandomForestRegressor

# Different hyperparameter values
hyperparameter_values = [0.01, 0.1, 0.5, 1.0]

# create experiment
my_exp = mlflow.set_experiment("new_experiment")

for lr in hyperparameter_values:
    with mlflow.start_run(experiment_id=my_exp.experiment_id):
        # log parameters
        mlflow.log_param("learning_rate", lr)

        # Add other hyperparameters and training logic
        model = RandomForestRegressor()
        mlflow.sklearn.log_model(model, name="model")


"""
Launch multiple experiments
While launching multiple runs within a single script provides a detailed view of variations in model configurations, MLflow also supports launching and managing multiple experiments.

This is particularly beneficial when dealing with diverse use cases, comparing different approaches, or conducting experiments across various datasets.

Let’s say in the same script, we may intend to experiment with different models (Logistic regression, Random Forest, etc.). For every model, we may have different hyperparameter values that we may want to test the model with.

In this case, each individual model type (Logistic regression, Random Forest, etc.) can be treated as an experiment. Furthermore, every configuration of the same model can be treated as a run.
"""
import mlflow

# Experiment 1) Logistic regression
log_reg = mlflow.set_experiment("logistic_reg")

with mlflow.start_run(experiment_id=log_reg.experiment_id):
    # Log parameters
    mlflow.log_param("algorithm", "logistic_regression")

    # Add more logging and training logic

# Experiment 2) Random forest model
rf = mlflow.set_experiment("rf_model")

with mlflow.start_run(experiment_id=rf.experiment_id):
    # Log parameters
    mlflow.log_param("algorithm", "Random forest")

    # Add more logging and training logic

## autologging

"""
# same code as earlier (load data, split etc.)

my_exp = mlflow.set_experiment("mlflow_autolog")

with mlflow.start_run(experiment_id=my_exp.experiment_id):

    mlflow.autolog()#enable auto logging

    # train model
    rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                      max_depth=max_depth)
    rf_model.fit(x_train, y_train)
        
    # generate predictions
    predictions = rf_model.predict(x_test)

    # determine performance metrics
    f1, accuracy, precision = performance(y_test, predictions)
"""