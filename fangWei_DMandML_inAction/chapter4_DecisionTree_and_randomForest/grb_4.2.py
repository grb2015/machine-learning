# ref : https://scikit-learn.org/stable/modules/tree.html
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
iris = load_iris()
X = iris['data']
y = iris['target']
print('x = ',X)
print('y = ',y)
# decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(X, y)
r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r)

# bg added  用训练好的模型进行预测
preX = [[5.9 ,3.0,  5.1, 1.8]]
# [[1, 1,1,1]]
rt = decision_tree.predict(preX)
print("rt = ",rt)
