import pandas as pd
from sklearn.datasets import make_classification

# data with 5 features and 1000 samples
X,y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

# display the shapes of x and y
print(f"shape of X: {X.shape}") # shape of x: (100,5)
print(f"shape of y: {y.shape}") # shape of y: (100,)

data = pd.DataFrame(X,columns=["f1","f2","f3","f4","f5"])
data["label"]=y

data.to_csv("data.csv",index=False)

