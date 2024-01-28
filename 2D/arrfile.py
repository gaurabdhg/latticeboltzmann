import numpy as np
import matplotlib.pyplot as plt

nx = 60
ny = 60
dataset = np.ones((300,100))
tmp = np.ones((nx,ny))

ci,cj=nx/2,ny/2
cr=15
I,J=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')

dist1=np.sqrt((I-ci)**2+(J-cj)**2)

tmp[np.where(dist1<cr)] = 0
dataset[10:70,20:80,]=tmp

plt.imshow(dataset)
plt.show()
plt.imsave('obj.png',dataset)