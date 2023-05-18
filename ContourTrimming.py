import cv2
import numpy as np
import imutils
import os
import statistics


img_mask = "C:/Users/SIU856511631/Documents/RearchAssistant/PradipCode/Mask.png"
img_path = "C:/Users/SIU856511631/Documents/RearchAssistant/DatasetSamples/set1sample5raw/set1sample5raw/set1sample5raw_0000.tif"
# mask_path = "C:/Users/SIU856511631/Documents/RearchAssistant/DatasetSamples/set1sample5raw/set1sample5rawMask/"
####################################
#
#    Find contours of an image
#       and mask background
#
#                by
#
#         Sandeep Goshika
#
####################################

# open source image file
image = cv2.imread(img_path)
image_cpy = image.copy()

with_contours = image.copy()

mask = cv2.imread(img_mask, cv2.IMREAD_GRAYSCALE)

mask_cpy = mask.copy()

# print(mask_cpy.shape)

    # convert image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find contours in the masked image
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Create a mask for the area outside the contours
mask_outside = np.zeros_like(mask_cpy)
cv2.drawContours(mask_outside, contours, -1, 255, cv2.FILLED)

# Change all pixels outside the contour to white
with_contours[mask_outside == 0] = (255, 255, 255)


# Convert image to blck and white
thresh, image_edges = cv2.threshold(with_contours, 160, 255, cv2.THRESH_BINARY)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((2, 2), np.uint8)
img_dilation = cv2.dilate(image_edges, kernel, iterations=1)

# Invert Image
mask_erosion = np.bitwise_not(img_dilation)

print(mask_erosion.shape)

img_gry = cv2.cvtColor(mask_erosion, cv2.COLOR_BGR2GRAY)

# Find Contours to resultant image
cnts = cv2.findContours(img_gry, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
rect_areas = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    rect_areas.append(w * h)
    avg_area = statistics.mean(rect_areas)
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cnt_area = w * h
    #Adjust the area of contopur to be captured
    if cnt_area < 50:
        img_gry[y:y + h, x:x + w] = 0
cv2.drawContours(image_cpy, cnts, -1, (0, 255, 0), 3)

# print(os.path.splitext(mask_path+filename)[0]+'.png')

# cv2.imwrite(os.path.splitext(mask_path+filename)[0]+'.png', mask_erosion)

# cv2.imshow('thresh', image_edges)
cv2.imshow('Dilation', img_gry)
cv2.imshow('original', image_cpy)


# escape condition
cv2.waitKey(0)

# clean up windows
cv2.destroyAllWindows()



















