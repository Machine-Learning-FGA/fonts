import numpy as np
import pandas as pd
from sklearn import tree

clf = tree.DecisionTreeClassifier()

## Reading csv files

candara = pd.read_csv("CANDARA.csv")
candara = candara.query("strength ==0.4 and italic==0.0 and orientation==0.0")

richard = pd.read_csv("RICHARD.csv")
richard = richard.query("strength ==0.4 and italic==0.0 and orientation==0.0")

## Dropping unnecessary features

candara.drop('fontVariant', axis=1, inplace=True)
candara.drop('strength', axis=1, inplace=True)
candara.drop('italic', axis=1, inplace=True)
candara.drop('orientation', axis=1, inplace=True)
candara.drop('h', axis=1, inplace=True)
candara.drop('w', axis=1, inplace=True)
candara['font'] = 0

richard.drop('fontVariant', axis=1, inplace=True)
richard.drop('strength', axis=1, inplace=True)
richard.drop('italic', axis=1, inplace=True)
richard.drop('orientation', axis=1, inplace=True)
richard.drop('h', axis=1, inplace=True)
richard.drop('w', axis=1, inplace=True)
richard['font'] = 1

both_files = pd.concat([candara, richard])
both_labels = np.concatenate((candara['font'].as_matrix(), richard['font'].as_matrix()))

first_candara_row = candara[:1]
first_richard_row = richard[:1]

clf.fit(both_files, both_labels)
print(clf.predict(first_candara_row))
print(clf.predict(first_richard_row))
