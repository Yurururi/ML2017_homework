# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:58:25 2017

@author: howard
"""
import os
import numpy as np
import numpy.linalg as la
import scipy.ndimage as pyi
import matplotlib.pyplot as plt


base_dir = os.path.dirname(os.path.realpath(__file__))
part_dir = os.path.join(base_dir,'part1')
imgs_dir = os.path.join(part_dir,'faceExpressionDatabase')
def read_faces_data():
    imgs = np.ndarray((64,64,100))
    f10sub = [chr(i) for i in range(ord('A'),ord('Z')+1)][:10]
    for l,apb in enumerate(f10sub):
        for k in range(10):
            img_name = '{}{:02d}.bmp'.format(apb,k)
            img_dir = os.path.join(imgs_dir,img_name)
            imgs[:,:,k+l*10] = pyi.imread(img_dir)
    return imgs
def mypca(featMat):
    featMat -= np.mean(featMat,0)
    featCov = np.dot(featMat,featMat.T)
    
    w,v = la.eigh(featCov)
    return np.real(w)[::-1],np.real(v)[:,::-1]
def reconstruct_face(mat,v,k=5):
    return np.dot(v[:,:k],np.dot(v[:,:k].T,mat))

def mRMSE(imgs,rec_imgs):
    return np.sqrt(np.mean((imgs-rec_imgs)**2))/256

imgs = read_faces_data()
mat = imgs.reshape((4096,100))

fig=plt.figure(figsize=(6,6))
plt.imshow(np.mean(imgs,axis=2),cmap='gray')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.tight_layout()
fig.savefig(os.path.join(base_dir,'average.png'))

w,v = mypca(mat)



fig=plt.figure(figsize=(6,6))
for k in range(9):
    ax = fig.add_subplot(3, 3, k+1)
    ax.imshow(v[:,k].reshape((64,64)),cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('{}'.format(k))
    plt.tight_layout()
fig.savefig(os.path.join(base_dir,'egnface.png'))

fig2=plt.figure(figsize=(10,10))
for k in range(100):
    ax = fig2.add_subplot(10, 10, k+1)
    ax.imshow(imgs[:,:,k],cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
fig2.savefig(os.path.join(base_dir,'faces.png'))


rec_imgs = reconstruct_face(mat,v)

fig3=plt.figure(figsize=(10,10))
for k in range(100):
    ax = fig3.add_subplot(10, 10, k+1)
    ax.imshow(rec_imgs[:,k].reshape((64,64)),cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
fig3.savefig(os.path.join(base_dir,'rec_faces.png'))

NN = 100
vRMSE = np.zeros((NN))
rec_imgs = np.zeros((4096,100))
for k in range(NN):
    tmp_img = reconstruct_face(mat,v[:,[k]],1)
    rec_imgs += tmp_img 
    vRMSE[k] = mRMSE(mat,rec_imgs)

fig4=plt.figure(figsize=(8,3))
plt.plot(vRMSE)
plt.plot([0,NN],[0.01,0.01],color='r')
plt.tight_layout()
fig4.savefig(os.path.join(base_dir,'err.png'))

print(np.where(vRMSE<0.01)[0][0])
