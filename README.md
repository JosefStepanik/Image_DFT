# Image_DFT
Codes for image inspection using Digital Fourier Transform for yield a resolution of image. 
Used references are :
https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/
https://github.com/codetoday-london/2D-Fourier-Transforms-In-Python

I consider, that all edges in grayscale image (e.g. transform white-black-white) represent frequencies in spatial domain. Therefore I computed fourier transform of image. Dots / values of pixels in frequency domain represent frequency components in original image. 
If length of original pixels 0,16 nm and size of image in pixels are determined, the real length of image is known too.
Spot blurring in diferent directions is not considered.

Script resolution.py reads all images from Images directory followed by computing resolution in each image due function from fourier_tool.py class. 

In this class I implemented functions for discrete fourier transform of square image with odd size. Other shapes are cutted. 