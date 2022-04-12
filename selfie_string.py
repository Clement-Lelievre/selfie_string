#https://datagenetics.com/blog/december12019/index.html
#The basic algorithm is a greedy one. At every move I calculate the ‘best’ next move (which pin to thread to), 
# make that move, account the effects of this thread to the contrast of the image, then repeat the process again making the
#  next best move from this new location. I keep doing this threading operation until the next thread I might make makes an
#  improvement to the image that is below a certain threshold

import cv2
from math import cos, sin, pi
from random import choice
import numpy as np
from utils import best_pin
from tqdm import tqdm
   
# path
path = 'nbs.png'
# Window name in which image is displayed
window_name = 'canvas'
   
image = cv2.imread(path)
#if image.shape[1] != image.shape[0]:
image = cv2.resize(image, (500,500))
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayImage = 255 - grayImage # invert colours so that black is 255

# Center coordinates
center_coordinates = (image.shape[0]//2, image.shape[1]//2)
 
# Radius of circle
radius = min(center_coordinates) - 5
  
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 1
  
# create the blank canvas
canvas = np.ones((image.shape[0], image.shape[1]))*255

# draw dots on the circle of the canvas, materializing the pins for the thread
nb_pins = 1_000
pins = []
for k in range(nb_pins): # create the pins
    dot_center_y = int(center_coordinates[1] + radius * sin(2*pi*k/nb_pins))
    dot_center_x = int(center_coordinates[0] + radius * cos(2*pi*k/nb_pins))
    canvas = cv2.circle(canvas, (dot_center_y, dot_center_x) , 4, color, thickness=-1)
    pins.append((dot_center_x, dot_center_y))

current_pin = choice(pins) # start with a random pin
position = grayImage # define the current canvas position
nb_iter = 1_000
contrast_last = 255
contrast_threshold = 30
nb_strings = 0
while contrast_last > contrast_threshold:
#for _ in tqdm(range(nb_iter)):
    destination_pin, destination_bresenham = best_pin(position, current_pin, pins=pins) # Select highest-scoring pin
    contrast_last = 0
    for pixel in destination_bresenham:
        contrast_last += position[pixel]
        position[pixel] //= 2 # Update position (meaning, halve all the pixel values of the pixels that the line just passed through on the grayscale image)
        canvas[pixel] = 0 # Draw black line on the canvas
        
    contrast_last /= len(destination_bresenham) # contrast_last is the average contrast of the line on the grayscale image
    current_pin = destination_pin
    nb_strings += 1
 
print(f'{nb_strings} strings were used')
# Displaying the canvas with the thread on it
cv2.imshow(window_name, canvas)
cv2.waitKey(100000)