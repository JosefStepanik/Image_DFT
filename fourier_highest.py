# fourier_highest.py

from os import CLD_STOPPED
from pickle import TRUE
from re import M
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt

image_filename = "Image_9.jpg"

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )

def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))

def display_plots(individual_grating, reconstruction, idx):
    plt.subplot(121)
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)

# Read and process image
image = plt.imread(image_filename)
# image = image[:, :, :3].mean(axis=2)  # Convert to grayscale

# Array dimensions (array is square) and centre pixel
# Use smallest of the dimensions and ensure it's odd
array_size = min(image.shape) - 1 + min(image.shape) % 2

# Crop image so it's a square image
image = image[:array_size, :array_size]
centre = int((array_size - 1) / 2)

# Get all coordinate pairs in the left half of the array,
# including the column at the centre of the array (which
# includes the centre pixel)
coords_left_half = (
    (x, y) for x in range(array_size) for y in range(centre+1)
    
)
# Sort points based on distance from centre
""" coords_left_half = sorted(
    coords_left_half,
    key=lambda x: calculate_distance_from_centre(x, centre),
    # reverse=True
) """

plt.set_cmap("gray")

ft = calculate_2dft(image)
scaled = np.log(abs(ft))
scaled_clone = np.copy(scaled)

max_freq = np.amax(scaled)
print(max_freq)
new_image = np.zeros(scaled.shape)

for x in scaled_clone:
    for y in x:
        if y < 11 :
            y = 0

max = 0
for index, value in np.ndenumerate(scaled_clone):
    #print(index_x, x)
    #find maximum
    if value > max :
        max = value
        #for normalization
    if value < 11 :
        new_image[index] = 0
    else :
        new_image[index] = value
        print(new_image, value)CLD_STOPPED






# Show grayscale image and its Fourier transform
plt.subplot(121)
plt.imshow(image)
plt.axis("off")
plt.subplot(122)
plt.imshow(scaled)
plt.axis("off")
plt.pause(2)

plt.show()