from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle as pkl

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)

with open('dummy_lr_model.pkl', 'wb') as f:
    pkl.dump(clf, f)
