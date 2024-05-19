import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.morphology import skeletonize
from skimage.util import invert

os.chdir("D:\Coding\Image_proccesing\OpticDisc")



def bgr2gray(color_image): 

    # Split the color image into its RGB channels
    B, G, R = cv2.split(color_image)
    
    # Perform the grayscale conversion using Craig's formula
    gray_image = 0.3 * R + 0.59 * G + 0.11 * B
    gray_image = gray_image.astype(np.uint8)  # Convert to uint8 (required for imshow)
    
    # Display the grayscale image
    cv2.namedWindow('GrayscaleImage', cv2.WINDOW_NORMAL)
    cv2.imshow("GrayscaleImage", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return gray_image


def morph(gray_image):
    kernel_open=10
    kernel_close=70
    # Define the structuring elements
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
    
    # Perform morphological opening to remove small noise
    image_opening = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel_opening)
    
    # Perform morphological closing to eliminate blood vessels
    image_closing = cv2.morphologyEx(image_opening, cv2.MORPH_CLOSE, kernel_closing)
    
    # Calculate the difference between the closing and original image
    image_difference = cv2.subtract(image_closing, gray_image)
    
    # Display the images (for visualization)
    cv2.namedWindow('Final Image (I4)', cv2.WINDOW_NORMAL)
    
    cv2.imshow("Final Image (I4)", image_difference)
    cv2.imwrite("final19.jpg", image_difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image_difference



def binariezed(morphed_image):
    binary_image = cv2.threshold(morphed_image, 2, 255, 
    cv2.THRESH_BINARY)[1]
    cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Binary Image",binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return binary_image



def connectedComponentAnalysis(binary_image):
 
    output_image = np.zeros(binary_image.shape, dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    # Iterate over each connected component
    for label in range(1, num_labels):
        # Exclude small components by area
        if stats[label, cv2.CC_STAT_AREA] >= 2000:
            output_image[labels == label] = 255
            
   
    cv2.namedWindow('Connected Components', cv2.WINDOW_NORMAL)
    cv2.imshow("Connected Components", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output_image

def findCentroid(only_vessel_image):
    # Calculate the moments of the binary image
    M = cv2.moments(only_vessel_image)
    
    # Calculate the centroid coordinates (Cx, Cy)
    Cx = int(M['m10'] / M['m00'])
    Cy = int(M['m01'] / M['m00'])
    
    # Print the centroid coordinates
    print("Centroid of the blood vessels: ({}, {})".format(Cx, Cy))
    
    # Draw the centroid on the image for visualization
    output_image = cv2.cvtColor(only_vessel_image, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_image, (Cx, Cy), 5, (0, 0, 255), -1)

    # Display the result
    cv2.namedWindow('Centroid of Blood Vessels', cv2.WINDOW_NORMAL)
    cv2.imshow('Centroid of Blood Vessels', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return Cx,Cy
    
    
def removeTemporal(Cx,filtered_image):
    # Determine the position of OD (left or right side)
    is_left_OD = Cx < filtered_image.shape[1] // 2

    # Draw a vertical reference line through the centroid
    reference_line = np.copy(filtered_image)
    cv2.line(reference_line, (Cx, 0), (Cx, filtered_image.shape[0]), (127), 1)
    
    # Remove vessels on the temporal side (opposite to OD) beyond the reference line
    if is_left_OD:
        filtered_image[:, :Cx] = 0
    else:
        filtered_image[:, Cx:] = 0
        
    cv2.namedWindow('Temporal Removed Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Temporal Removed Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return filtered_image

def skeleton(temporal_removed_image):
    # Skeletonize the remaining vessels
    skeleton= cv2.ximgproc.thinning(temporal_removed_image)
    ##binary_image_inverted = cv2.bitwise_not(temporal_removed_image)
    ## binary_image_bool = binary_image_inverted.astype(np.bool_)
    ## skeleton = skeletonize(binary_image_bool)
    ##skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    ##skeleton_final = cv2.bitwise_not(skeleton_uint8)
    cv2.namedWindow('Skeletonized and Pruned', cv2.WINDOW_NORMAL)
    cv2.imshow('Skeletonized and Pruned', skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# Example usage
image = cv2.imread("fundus_image.jpg")
gray_image = bgr2gray(image)
morphed_image = morph(gray_image)
binariezed_image = binariezed(morphed_image)
only_vessel_image = connectedComponentAnalysis(binariezed_image)
Cx, Cy = findCentroid(only_vessel_image)
temporal_removed_image = removeTemporal(Cx, only_vessel_image)
skeleton(temporal_removed_image)

cv2.imwrite("BinaryImage.jpg", binariezed_image)
cv2.imwrite("Connected.jpg",only_vessel_image)









