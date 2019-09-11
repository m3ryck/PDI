#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:23:10 2019

@author: adriano & alexandre & arthur & olivio & tgg
"""
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import skimage
from skimage import data, io
import math
from math import cos, sqrt, pi
import os
from scipy.fftpack import dct, idct
from math import cos, pi, sqrt


img_original = image.imread('/home/adriano/Documentos/arquivos pdi/trabalho 2/lena.bmp')

img = img_original.copy()

img


os.chdir("/home/adriano/Documentos/arquivos pdi/trabalho 2")

x=[4 , 3 , 5 , 10]
N=4

#from scipy.io import wavfile
#import numpy as np
#fs , data = wavfile.read('MaisUmaSemana.wav')
#xi = data.copy()
#x = np.array(xi)
#N = x.size
#

#X = x.copy()


def IDCT(X,N):    
    x = np.zeros(N)
    ck = ((1/2)**(1/2))
    for n in range(0,N-1):
        somatorio = 0
        for k in range(0,N-1):
            somatorio +=  ck * X[k]*math.cos(((2*(3.141592653589)*k*n)/2*N) + ((k*(3.141592653589))/2*N))
        x[n] = ((2/N)**(1/2)) * somatorio        
        ck=1
    return x


def DCT(x,N):
    X =  np.zeros_like(x).astype(float)
    for k in range(0,N-1):
        somatorio = 0
        for n in range(0,N-1):
            somatorio += x[n]* cos(2*pi*k/(2.0*N)*n + (k*pi)/(2.0*N))
        ck = sqrt(0.5) if k == 0 else 1
        X[k] = sqrt(2.0 / N) * ck * somatorio        
    return X

def DCT2D(img,N):    
    n=N
    altura = img.shape[0]
    largura = img.shape[1]    
    imgLinha = np.zeros_like(img).astype(float)
    imgColuna = np.zeros_like(img).astype(float)
    
    for i in range(altura):
        imgLinha[i,:] = DCT(img[i,:],n)
        
    for j in range(largura):
        imgColuna[:,j] = DCT(imgLinha[:,j],n)
    return imgColuna

tam = len(img)
tam

dct2 = DCT2D(img,tam)
print(dct2)


plt.imshow(dct2,'gray')
dct2shift = np.fft.fftshift(dct2)

from matplotlib.colors import LogNorm
plt.imshow(dct2shift, 'gray')

plt.imshow(dct2)







asd=DCT(x,N)
print(asd)

dsa=IDCT(asd,N)/6
print(dsa)

asd=dct_1d(x,N)
print(asd)

dsa=IDCT(asd,N)/6
print(dsa)



teste = dct(x,type = 2)
print(teste)

teste2 = idct(teste,1)/6
print(teste2)




               


























