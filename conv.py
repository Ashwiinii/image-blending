import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def padding(img, mode, pad_width):
    width, height = img.shape[1], img.shape[0]
    # img_array = np.array(img)
    img_array = img
    m = pad_width
    n = pad_width
    new_height = int(height + 2*(m))
    new_width = int(width + 2*(n))
    
    if mode == 'zero':
        zero_padding = np.zeros((new_height, new_width))   
        for i in range(0, width):
            for j in range(0, height):
                zero_padding[j+m, i+n] = img_array[j,i]
                
        padding_image = zero_padding
                
        
    elif mode == 'wrap':
        wrap_padding = np.zeros((new_height, new_width))
        
        for i in range(0, width):
            for j in range(0, height):
                wrap_padding[j+m, i+n] = img_array[j,i]

        wrap_padding[0 : m, : ] = wrap_padding[-1*(2*m) : m + height, : ]
        wrap_padding[-1*m : , : ] =wrap_padding[m : 2*m, :]
        wrap_padding[ : ,-1*(n) : ] = wrap_padding[ : ,n : 2*n]
        wrap_padding[ : ,0 : n] = wrap_padding[ : ,-1*(2*n) : n + width]
            
        padding_image = wrap_padding  
        
    elif mode == 'copy':
        copy_padding = np.zeros((new_height, new_width))
            
        for i in range(0, width):
            for j in range(0, height):
                copy_padding[j+m, i+n] = img_array[j,i]
      

        copy_padding[0 : m, : ] = copy_padding[[m], : ]
        copy_padding[-1*(m) : , : ] = copy_padding[[-1*m-1], :]
        copy_padding[ : ,-1*n : ] = copy_padding[ : ,[-1*n-1]]
        copy_padding[ : ,0 : n] = copy_padding[ : ,[n]]

        padding_image = copy_padding

        
    elif mode == 'reflect':
        reflect_padding = np.zeros((new_height, new_width), dtype=np.float32)  
        # print(reflect_padding.shape)
        
        for i in range(0, width):
            for j in range(0, height):
                reflect_padding[j+m, i+n] = img_array[j,i]
                

        reflect_padding[0 : m, : ] = np.flip(reflect_padding[m : 2*m, :],axis = 0)
        reflect_padding[-1*(m) : , : ] = np.flip(reflect_padding[-2*(m) : -1*(m), : ], axis = 0)
        reflect_padding[ : ,-1*(n) : ] = np.flip(reflect_padding[ : ,-2*(n) : -1*(n)], axis = 1)
        reflect_padding[ : ,0 : n] = np.flip(reflect_padding[ : ,n : 2* n], axis = 1)
            
        padding_image = reflect_padding
    return padding_image

#convolution
def my_split(img: np.ndarray):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    return b,g,r

def conv(image, w, pad_mode='reflect'):
    padded_image = padding(image, pad_mode, 2)
    
    conv_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)       #for stride 1
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            n_matrix = padded_image[i:i + w.shape[0], j:j + w.shape[1]]

            conv_value = np.sum(np.multiply(n_matrix, w))
            conv_image[i][j] = conv_value
    
    return conv_image

def myfft2(img: np.ndarray,m,n):
    fft2 = np.fft.fft(np.fft.fft(img, m,axis=0), n,axis = 1)
    return fft2

# Inverse FFT2
def myifft2(fft: np.ndarray,m,n):
    ifft2 = np.conj(myfft2(np.conj(fft),m,n))
    ifft2 = np.real(ifft2)/((fft.shape[0]*fft.shape[1]))
    return ifft2

def convolve_fft(c: np.array, kernel:np.array) -> np.array:

    c = padding(c, "reflect", 2)
    # kernel = padding(kernel, 'reflect', 2)
    output_shape = (c.shape[0], c.shape[1])

    fft_kernel = myfft2(kernel, output_shape[0]+2, output_shape[1]+2)
    # fft_kernel = np.fft.fft2(kernel, output_shape[0], output_shape[1])
    # fft_img = np.fft.fft2(c, output_shape[0], output_shape[1])
    fft_img = myfft2(c, output_shape[0]+2, output_shape[1]+2)

    mul = fft_img*fft_kernel

    out = myifft2(mul, output_shape[0]-4, output_shape[1]-4)

    return np.real(out)


def conv2(image, w, pad_mode='reflect'):
    
    # if kernel == 'box':
    #     w = 1/9*np.ones((3,3))
    # elif kernel == 'first order derivative row':
    #     w = np.array([[-1,1,0], [0, 0, 0], [0, 0, 0]])   #[-1, 1]
    # elif kernel == 'first order derivative column':
    #     w = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0]])  #[[-1], [1]]
    # elif kernel == 'prewitt mx':
    #     w = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # elif kernel == 'prewitt my':
    #     w = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    # elif kernel == 'sobel mx':
    #     w = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # elif kernel == 'sobel my':
    #     w = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    # elif kernel == 'robert mx':
    #     w = np.array([[0,1, 0],[-1,0, 0], [0, 0, 0]])   #[[0,1],[-1,0]]
    # elif kernel == 'robert my':
    #     w = np.array([[1,0, 0],[0,-1, 0], [0, 0, 0]])     #[[1,0],[0,-1]]
        
    b,g,r = my_split(image)

    conv_b = conv(b, w)
    conv_g = conv(g, w)
    conv_r = conv(r, w)

    # conv_b = convolve_fft(b, w)
    # conv_g = convolve_fft(g, w)
    # conv_r = convolve_fft(r, w)

    # result = Image.merge('RGB', (conv_r, conv_g, conv_b))
    result = np.dstack((conv_b,conv_g,conv_r))
    
    return result
