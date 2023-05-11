import numpy as np
import cv2
import pyramid_functions
from conv import conv2

def blend(l_fimg, l_bimg, gMask, layers):
    
    gMask.reverse()
    
    LS = []
    for la,lb,mask in zip(l_fimg,l_bimg,gMask):
        ls = la * mask + lb * (1 - mask)
        LS.append(np.float32(ls))

    lap_bl = LS[0]
    for i in range(1,layers):
        lap_bl = conv2(pyramid_functions.upSampler(lap_bl,LS[i].shape),w)
        lap_bl = lap_bl.astype(np.float32)
        lap_bl = cv2.add(lap_bl,LS[i])
        
    final = np.clip(lap_bl,0,255).astype(np.uint8)
    
    return final


def create_mask(aligned_fg_img, init_x, init_y, x1, y1):
    
    new_mask = np.zeros(aligned_fg_img.shape).astype(np.float32)
    
    
    if roi == "ellipse":
        new_mask = cv2.ellipse(new_mask, ((init_x+(x1[0]-init_x)//2),(init_y+(y1[0]-init_y)//2)), (x1[0]-init_x,y1[0]-init_y), 0,0,360, (1,1,1), -1)
    elif roi == "rectangle":
        new_mask = cv2.rectangle(new_mask, (init_x,init_y),(x1[0],y1[0]), (1,1,1), -1)
    else:
        points = np.column_stack((x1, y1))
        print(points)
        cv2.fillPoly(new_mask, [points], (1, 1, 1))

    return new_mask 

#manually aligning the foreground image with background image 
    
def align(f_img,b_img,row,col):
    
    a_img = np.zeros(b_img.shape)                  
    a_img[row:(f_img.shape[0]+row) , col:(f_img.shape[1]+col)] = f_img         
   
    return a_img.astype(np.uint8)                                 


#to mark rectangle, ellipse or free hand roi
def get_mask_shape(event,x,y,f,param):                    

    global init_x, init_y, x1, y1, mouse_pressed, points 
    
    if event == cv2.EVENT_LBUTTONDOWN:            
        mouse_pressed = True
        init_x,init_y = x,y                       
        
    elif event == cv2.EVENT_LBUTTONUP:            
        mouse_pressed = False
        
        if roi == "ellipse":
            cv2.ellipse(aligned_fg_img, ((init_x+(x-init_x)//2),(init_y+(y-init_y)//2)), (x-init_x,y-init_y), 0,0,360, (0,0,0), 1)
        elif roi == "rectangle":
            cv2.rectangle(aligned_fg_img, (init_x,init_y), (x,y), (0,0,0), 1)     
        else:
            cv2.circle(aligned_fg_img, (init_x, init_y), 3, (0, 0, 255), 3, cv2.FILLED)
            points.append((init_x, init_y))
            if len(points)>=2:
                cv2.line(aligned_fg_img, (points[len(points)-1]), (points[len(points)-2]),(0,0,0),5)

            x1.append(x)
            y1.append(y)

           
            
        #store last positions of x and y
        if roi == 'ellipse' or roi == 'rectangle':                              
            x1.append(x)
            y1.append(y)

mouse_pressed = False   #flag variabl turns true when mouse is pressed
roi = ""                #region of interest (ellipse, rect and free hand)
init_x,init_y = -1,-1  #start points of x and y (roi)
points = []
x1 = []
y1 = []


b_img = cv2.resize(cv2.imread("mona_lisa.jpg"), (650,900))    #read backgroud image         
f_img = cv2.imread("face_try.jpg")                             #read foreground image
 


if (b_img.shape[0] > f_img.shape[0]):
    f_img_align = align(f_img,b_img,30,120)
    f_img = np.copy(f_img_align)
    aligned_fg_img = np.copy(f_img_align)
else:
    aligned_fg_img = np.copy(f_img)



cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Foreground Image', get_mask_shape)      #enable GUI to draw ROI

while(1):
    cv2.imshow('Foreground Image',aligned_fg_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):      #press e on keyboard to draw ellipse                       
        roi = 'ellipse'                   
    elif key == ord('r'):
        roi = 'rectangle'    #press r on keyboard to draw rectangle  
    elif key == 27:                               
        cv2.destroyAllWindows()
        break
    

mask = create_mask(aligned_fg_img, init_x, init_y, x1, y1)



cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.imshow('Foreground Image',f_img)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.imshow('Mask', mask)
cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
cv2.imshow('Background Image', b_img)
cv2.waitKey(0)                                     
cv2.destroyAllWindows()

# Laplacian pyramid for foreground image, background image and Gaussian mask

print("Pyramids for foreground image")
f_img_gPyr, f_img_lPyr = pyramid_functions.ComputePyr(f_img,10)
print("Pyramids for background image")
b_img_gPyr, b_img_lPyr = pyramid_functions.ComputePyr(b_img,10)

print("Pyramids for Gaussian mask")
g_mask = np.copy(mask).astype(np.float32)
mask_gPyr = [g_mask]
g_kernel = cv2.getGaussianKernel(5,2)
w = g_kernel*(g_kernel.T)
for i in range(len(f_img_gPyr)-1):
    g_mask = pyramid_functions.downSampler(conv2(g_mask,w))
    mask_gPyr.append(g_mask)

print("Blending in process!!")
blended_img = blend(f_img_lPyr, b_img_lPyr, mask_gPyr, len(f_img_gPyr))

# Display all the images

cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.imshow('Foreground Image',aligned_fg_img)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.imshow('Mask',mask)
cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
cv2.imshow('Background Image',b_img)
cv2.namedWindow('Blended Image', cv2.WINDOW_NORMAL)
cv2.imshow('Blended Image',blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite('aligned_fg_img.png', aligned_fg_img)
cv2.imwrite('b_img.png', b_img)
cv2.imwrite('bl_img.png', blended_img)
