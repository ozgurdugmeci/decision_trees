import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


pathy="C:\\exceller\\trees\\trees.xlsx"
trees=pd.read_excel(pathy, 'Sheet1')


X = trees['ktsy']
Y = trees['clsfr']

X.columns= ['ktsy']




X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

print(len(trees))
print(len(X_train))


X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

X_train=X_train.reshape(1219, 1)
Y_train= Y_train.reshape(1219, 1)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
plt.figure(figsize=(8,8))
tree.plot_tree(clf.fit(X_train, Y_train),class_names=['3 ay uyum','4 ay uyum','yeni urun_kötü satış','yeni ürün_aksak satıs'],filled=True) 

plt.show()							 


