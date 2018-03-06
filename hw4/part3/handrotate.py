import numpy as np
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues
import scipy.ndimage as pyi
import matplotlib.pyplot as plt
import os
base_dir = os.path.dirname(os.path.realpath(__file__))
imgs_dir = os.path.join(base_dir,'hand')
def read_hands_data():
    imgs = np.ndarray((480,512,481))
    for k in np.arange(481):
        img_name = 'hand.seq{}.png'.format(k+1)
        img_dir = os.path.join(imgs_dir,img_name)
        imgs[:,:,k] = pyi.imread(img_dir)
    return imgs
imgs = read_hands_data()
resize_rate = 16
small_imgs = imgs[::resize_rate,::resize_rate,:]
feats = (small_imgs.reshape((int(np.size(small_imgs)/481),481))).T


vs = get_eigenvalues(feats,NEIGHBOR=30,SAMPLE=20)
#plt.plot(vs[:30].T)

npzfile = np.load('large_data.npz')
X = npzfile['X']
y = npzfile['y']
SSVR = SVR(C=0.6)
SSVR.fit(X,y)
pred_y = SSVR.predict(vs.flatten()[:100])

