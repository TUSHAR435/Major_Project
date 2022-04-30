import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
#shape
print(dataset.shape)
#head
print(dataset.head(20))
#describe
print(dataset.describe())
#distribution(class)
print(dataset.groupby('class').size())
#histogram
dataset.hist()
plt.show()
#scatter plot matrix
scatter_matrix(dataset)
plt.show()
#split the dataset
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
validation_size=0.20
seed=7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

models=[]
models.append(('LR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('SNM',SVC()))
print(models)

# evaluate each model in turn
results = []
names = []

for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cvresults = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold, scoring=scoring)
    results.append(cvresults)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cvresults.mean(),cvresults.std())
    print(msg)

# Make predictions on validation dataset

for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(name)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
