import cv2
import numpy as np
import matplotlib.pyplot as plt

# import image to be used
pic = cv2.imread('img/night1.jpg')
# convert from BGR to RGB
rgb_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_pic)

# import greenscreen background picture
pic_green = cv2.imread('img/green_screen.png')
rgb_green = cv2.cvtColor(pic_green, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_green)

print(pic.dtype, pic.shape)
print(pic_green.dtype, pic_green.shape)

# resize both pictures to be the same size
normal = cv2.resize(rgb_pic,(298,168))
green = cv2.resize(rgb_green,(298,168))

#show hsv colour map so we can extract ranges for colour green
hsv_map = cv2.imread('img/hsv.png')
plt.imshow(hsv_map)

#for green colours the upper and lower bounds on hsv colour map are
lower_bound = (38,40, 20)
upper_bound = (85, 255,255)

# convert green image from colour to hsv and mask it to green colour range
hsv_image = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_image, lower_bound,upper_bound)

# returns image with all colours not green as white, i.e. the background
masked = green.copy()
masked[mask!=0] = (255, 255, 255)
plt.imshow(masked)

# returns green image contents with first picture background and not a green background
final_image = cv2.bitwise_and(normal,masked)
plt.imshow(final_image)