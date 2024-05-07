
from sklearn import svm
from sklearn import datasets
from zutils import save_model,load_model

clf=svm.SVC()
iris = datasets.load_iris()
X,y = iris.data,iris.target
clf.fit(X,y)
# save_model("test",clf)
clf1=load_model("test")
print(clf1.predict(X[0:3,:]))


