import pandas as pd
from sklearn.metrics import f1_score, accuracy_score , precision_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

test_size=0.2
n_estimators=40
max_depth=7

def performance(actual,pred):
    f1 = f1_score(actual,pred)
    accuracy=accuracy_score(actual,pred)
    precision=precision_score(actual,pred)
    return f1,accuracy,precision

if __name__=="__main__":
    # read the data csv file
    data = pd.read_csv("data.csv")
    # split the data into training and test sets
    train,test = train_test_split(data,test_size=test_size)

    # create (X,y) data
    x_train = train.drop(["label"],axis=1)
    y_train = train["label"]

    x_test = test.drop(["label"],axis=1)
    y_test = test["label"]

    # train model
    rf_model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf_model.fit(x_train,y_train)

    # generate predictions
    predictions = rf_model.predict(x_test)

    # determine the performance metrics
    f1,accuracy,precision = performance(y_test,predictions)

    #print model details
    print(f"Random Forest: {n_estimators=}, {max_depth=}")
    print(f"F1 score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

"""
- start by reading csv file
- split the data into training and test sets based on specified test_size parameter
- next as data has all columns including label, we create in X,y format for training the model
- we instantiate a RandomForestClassifier model with specified parameters earlier and train the model
- after training,we generate predictions on test data using predict() method and evaluate them using the performance() method we defined earlier
- finally we print the model details such as hyperparameters,accuracy,F1 and precision
"""