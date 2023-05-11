import numpy as np
import cv2
from conv import conv2

# function to down sample and to get the upper layer
def downSampler(img):
    
    row, col = img.shape[0], img.shape[1]
    down_row = (np.floor(row/2)).astype(np.int16)
    down_col = (np.floor(col/2)).astype(np.int16)
    
    row_ratio = down_row/row
    col_ratio = down_col/col
    
    select_rows = (np.floor((np.arange(0,down_row,1))/row_ratio)).astype(np.int16)
    select_cols = (np.floor((np.arange(0,down_col,1))/col_ratio)).astype(np.int16)

    output = img[select_rows , :]
    output = output[: , select_cols]
    
    return output

# function to up sample and to reconstruct lower layer
def upSampler(img,dstsize):
    
    row, col = img.shape[0], img.shape[1]

    up_row, up_col = dstsize[0], dstsize[1]
    
    row_ratio = up_row/row
    col_ratio = up_col/col
    
    select_rows = (np.floor((np.arange(0,up_row,1))/row_ratio)).astype(np.int16)
    select_cols = (np.floor((np.arange(0,up_col,1))/col_ratio)).astype(np.int16)
    
    output = img[select_rows , :]
    output = output[: , select_cols]

    return output 

# function to compute the pyramids
def ComputePyr(img,num_layers):

    shape = min(img.shape[0], img.shape[1])

    max_possible_layers = int(1 + np.floor(np.log2(shape/5)))
    pyramid_layers = min(num_layers, max_possible_layers)
    
    # 1d Gaussian kernel with std deviation 2
    g_kernel = cv2.getGaussianKernel(5,2)           
    w = g_kernel*(g_kernel.T) 
    
    # Gaussian Pyramid
    lower = img.copy().astype(np.float32)
    g_pyr = [lower]
    for i in range(pyramid_layers-1):
        lower = downSampler(conv2(lower,w))
        g_pyr.append(lower)

    # Laplacian Pyramid
    l_pyr = [g_pyr[pyramid_layers-1]]
    for i in range(pyramid_layers-1,0,-1):
        dstsize = g_pyr[i-1].shape

        gaussian_expanded = conv2(upSampler(g_pyr[i],dstsize),w)
        laplacian = np.subtract(g_pyr[i-1], gaussian_expanded)
        
        l_pyr.append(laplacian)
    return g_pyr, l_pyr


