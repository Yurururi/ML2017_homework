import numpy as np
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues
import time
import sys
# Train a linear SVR
stt = time.time()

args = sys.argv
npzfile = np.load('large_data.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)
data_size = np.array([10000, 20000, 50000, 80000, 100000])
svr = [[]]*5
ss = int(np.size(X,0)/5)
for k in range(5):
    svr[k] = SVR(C=0.6)
    svr[k].fit(X[ss*k:ss*(k+1),:], np.log(y[ss*k:ss*(k+1)]))
# svr.get_params() to save the parameters
# svr.set_params() to restore the parameters

# predict
testdata = np.load(args[1])
test_X = np.zeros((200,100))
testdata_size_idx = np.zeros((200,))
for i in range(200):
#    print('round {}...'.format(i))
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    test_X[i,:] = vs
    testdata_size_idx[i] = np.where(data_size==np.size(data,0))[0]
test_X = np.array(test_X)
pred_y = np.zeros((200,))
for k in range(5):
    idx = np.where(testdata_size_idx==k)[0]
    pred_y[idx] = svr[k].predict(test_X[idx,:])
pred_y=np.round(np.exp(pred_y))
pred_y[pred_y==0]=1
pred_y = np.log(pred_y)


with open(args[2], 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
        print('{},{}'.format(i,d), file=f)
#print('Total time:{}sec'.format(time.time()-stt))