'''
color train
color collect
color train
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from multiprocessing import Pool,cpu_count
import pickle

dataset = pd.read_csv('../data/data.csv')
le = LabelEncoder()
le.fit(dataset['color'])
train = dataset[['r','g','b']]
labels = le.transform(dataset['color'])

sss = StratifiedShuffleSplit(n_splits = 10, test_size=0.2, random_state=23)
X_train, X_test, y_train, y_test = None,None,None,None
for train_index, test_index in sss.split(train,labels):
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

def classifer_comp(clf):
    name = clf.__class__.__name__
    print(name)
    clf.fit(X_train, y_train)

    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)

    print("=" * 30)
    print(name)
    print('****Results****')
    print("Accuracy: {:.4%}".format(acc))
    print("Log Loss: {}".format(ll))
    print("=" * 30)

def classifer_comp_multiprocess():
    p = Pool(cpu_count())
    p.map(classifer_comp,classifiers)

def run():
    '''
    choose RandomForestClassifier
    :return:
    '''
    clf = RandomForestClassifier()
    clf.fit(train,labels)
    model = '../model/di_model.pkl'
    pickle.dump(clf,open(model,'wb'))

    return


if __name__ == '__main__':
    run()