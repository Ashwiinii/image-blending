# image-blending

Pyramid blending refers to seamlessly blending an area/region from a source(foreground) image into a target (background) image. This project develops functions to implement Gaussian/Laplacian pyramid.

The project is divided into 2 major parts:
- A GUI to create a mask
- Gaussian/Laplacian blending

## Steps:
### Image Masking:
  - Align foreground and background images.
  - The GUI can generate a black/white image, called the mask, of the same size as the opened image, in which the selected region(s) are white and the remaining black.
  - The GUI to create mask on the foreground image is able to create masks of following shapes:
    - ellipse
    - rectangle
    - free shape
### Blending:
  - Generate gaussian pyramid of both foreground and background image.
  - Generate laplacian pyramid of both images of gaussian (fL, bL).
  - Generate gaussian pyramid of the mask (mG).
  - Generate blended image using fL, bL, and mG.


## Example images

An example foreground-background pair. 
<p>
    <img src="test_images/Pair 1/foreground.jpg" width="300" alt="Foreground Image">
    <img src="test_images/Pair 1/background.png" width="300" alt="Background Image">
</p>

Final blended image.
<p>
    <img src="test_images/Pair 1/blendedimg.png" width="300" alt="Blended Image">
</p>


## Code description

- [run.py](https://github.com/Ashwiinii/image-blending/blob/main/run.py): The main file, run this file to get the blended image
- [conv.py](https://github.com/Ashwiinii/image-blending/blob/main/conv.py): Contains convolution code and padding code 
- [pyramid _functions.py](https://github.com/Ashwiinii/image-blending/blob/main/pyramid_functions.py): Function for computing Gaussian and Laplacian pyramid
