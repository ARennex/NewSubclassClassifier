print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from .. import PRF
from sklearn.ensemble import RandomForestClassifier
import numpy


import os
folder = os.getcwd() + "/PRF/examples/data"

#folder = 'D:/NewSubclassClassifier/NewSubclassClassifier/Useful Code/PRF/PRF/examples/data'
X = numpy.load(folder+'/bootstrap_X.npy')
y = numpy.load(folder+'/bootstrap_y.npy')
dX = numpy.load(folder+'/bootstrap_dX.npy')
# X = numpy.load('data/bootstrap_X.npy')
# y = numpy.load('data/bootstrap_y.npy')
y[y > 2] = 2

n_objects = X.shape[0]
n_features = X.shape[1]
print(n_objects, 'objects,', n_features, 'features')

shuffled_inds = numpy.random.choice(numpy.arange(n_objects),n_objects,replace=False)

shuffled_inds = numpy.where( (y == 1)  |  (y == 2) |  (y == 4)|  (y == 5)|  (y == 6)|  (y == 8)|  (y == 13))[0]
shuffled_inds = numpy.random.choice(shuffled_inds,len(shuffled_inds),replace=False)
n_train = 5000
n_test = 500
print('Train set size = {}, Test set size = {}'.format(n_train, n_test))

nf = n_features
train_inds = shuffled_inds[:n_train]
X_train = X[train_inds][:,:nf]
y_train = y[train_inds]
dX_train = dX[train_inds][:,:nf]

test_inds = shuffled_inds[n_train:(n_train + n_test)]
X_test = X[test_inds][:,:nf]
y_test = y[test_inds]
dX_test = dX[test_inds][:,:nf]

#import faulthandler
#faulthandler.enable()

print('Break Test 1')

n_trees = 10
#n_trees = 100
prf_cls = PRF.RandomForestClassifier(n_estimators=n_trees,  bootstrap=True)
#prf_cls = PRF.prf(n_estimators=n_trees,  bootstrap=True)

print('Break Test 2')

prf_cls.fit(X=X_train, y=y_train, dX=dX_train)
#prf_cls.fit(X=X_train, y=y_train)

print('Break Test 3')

print(prf_cls.score(X_test, y=y_test, dX=dX_test))
#print(prf_cls.score(X_test, y=y_test))
