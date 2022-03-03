# fourier_tool.py
# Class for compute resolution value in imported image. 
# Image is cutted to square image with odd size values 

import numpy as np
import matplotlib.pyplot as plt

class fourier_tool: 
    @staticmethod
    def _calculate_2dft(input):
        ft = np.fft.ifftshift(input)
        ft = np.fft.fft2(ft)
        return np.fft.fftshift(ft)
    @staticmethod
    def _calculate_distance_from_centre(coords, centre):
        # Distance from centre is √(x^2 + y^2)
        return np.sqrt((coords[0] - centre) ** 2 + (coords[1] - centre) ** 2)
    @staticmethod
    def fourier_inspect(image_file):
        # Treshold value for cutting other useless frequencies
        tresh = 11 
        # Dimension of one pixel in nanometers
        pixel_size = 0.16 
        # Read and process image
        image = plt.imread(image_file)
        # Array dimensions (array is square) and centre pixel
        # Use smallest of the dimensions and ensure it's odd
        array_size = min(image.shape) - 1 + min(image.shape) % 2
        # Crop image so it's a square image
        image = image[:array_size, :array_size]
        centre = int((array_size - 1) / 2)
        # Length of side in nm 
        side_length = array_size*pixel_size
        # Calculate fourier transform
        ft = fourier_tool._calculate_2dft(image)
        # After fourier transform we get complex number. Therefore we consider absolute values and log scale for grayscale view
        scaled = np.log(abs(ft))
        new_image = np.zeros(scaled.shape)
        map_image = np.zeros(scaled.shape)
        # Iteration for normalize with treshold value
        for index, value in np.ndenumerate(scaled):
            # for normalization
            if value < tresh :
                new_image[index] = 0
            else :
                new_image[index] = value
                map_image[index] = fourier_tool._calculate_distance_from_centre(index, centre)
        # In frequency space is image symectrical to Y axis, we assume can assume only half
        map_image_half = map_image[0:int((array_size - 1) / 2),:]
        # Find max value and calculate resolution
        freq = np.amax(map_image_half)
        resolution = round(side_length/freq,2)
        return resolution