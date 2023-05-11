# image-blending

Pyramid blending refers to seamlessly blending an area/region from a source(foreground) image into a target (background) image. This project develops functions to implement Gaussian/Laplacian pyramid.

The project is divided into 2 major parts:
- A GUI to create a mask
- Gaussian/Laplacian blending

## Steps:
### Image Masking:
  - Align foreground and background images.
  - The GUI can generate a black/white image, called the mask, of the same size as the opened image, in which the selected region(s) are white and the remaining black.
  - GUI to create mask on the foreground image of following shapes:
    - ellipse
    - rectangle
    - free shape
### Blending:
  - Generate gaussian pyramid of both foreground and background image.
  - Generate laplacian pyramid of both images of gaussian (fL, bL).
  - Generate gaussian pyramid of the mask (mG).
  - Generate blended image using fL, bL, and mG.

## Code description

- run.py: The main file, run this file to get the blended image
- conv.py: Contains convolution code and padding code 
- pyramid _functions.py: Function for computing Gaussian and Laplacian pyramid
