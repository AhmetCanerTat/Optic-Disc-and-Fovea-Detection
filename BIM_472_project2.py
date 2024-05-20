import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from skimage import morphology


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
        reference_line[:,Cx :] = 0
    else:
        reference_line[:, :Cx] = 0
        
    cv2.namedWindow('Temporal Removed Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Temporal Removed Image', reference_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return reference_line

def skeleton(temporal_removed_image):
    # Skeletonize the remaining vessels
    ##skeleton= cv2.ximgproc.thinning(temporal_removed_image)
    temporal_removed_bool=temporal_removed_image.astype(bool)
    skeleton_bool=morphology.thin(temporal_removed_bool,11)
    skeleton = (skeleton_bool * 255).astype(np.uint8)
    cv2.namedWindow('Skeletonized', cv2.WINDOW_NORMAL)
    cv2.imshow('Skeletonized', skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return skeleton
    
    
def prune(skeleton):
    ##pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skeleton, size=100)
    skeleton_bool = skeleton.astype(bool)
    pruned_bool= morphology.remove_small_objects(skeleton_bool,50,connectivity=1)
    pruned_skeleton = (pruned_bool * 255).astype(np.uint8)
    cv2.namedWindow('Pruned', cv2.WINDOW_NORMAL)
    cv2.imshow('Pruned', pruned_skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pruned_skeleton

def findIntersection(pruned_skeleton):
    horizontal_line = np.copy(pruned_skeleton)
    cv2.line(horizontal_line, (0, centroid[1]), (horizontal_line.shape[1], centroid[1]), (127), 1)
    intersection = cv2.bitwise_and(horizontal_line, pruned_skeleton)
    intersection_points = np.argwhere(intersection == 127)
    output_image = cv2.cvtColor(horizontal_line, cv2.COLOR_GRAY2BGR)
    # Assuming the intersection point closest to the centroid is the OD center
    if intersection_points.size > 0:
        OD_center = intersection_points[np.argmin(np.linalg.norm(intersection_points - np.array([centroid[1], centroid[0]]), axis=1))]
        cv2.circle(output_image, (OD_center[1], OD_center[0]), 5, (0, 0, 255), -1)
    else:
        OD_center = (centroid[1], centroid[0])
      
    cv2.namedWindow('Intersection', cv2.WINDOW_NORMAL)
    cv2.imshow('Intersection', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return OD_center

def watershed(image,OD_center):
    center = (int(OD_center[1]),int(OD_center[0]))
    output = np.copy(image)
    gray_image = bgr2gray(image)
    ret, bin_img = cv2.threshold(gray_image,
                             100, 255, 
                             cv2.THRESH_BINARY )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, 
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=5)
    
    # sure background area
    sure_bg = cv2.dilate(bin_img, kernel, iterations=5)
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)  
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    
    # watershed Algorithm
    markers = cv2.watershed(image, markers)
    labels = np.unique(markers)
 
    od = []
    for label in labels[2:]:  
     
        # Create a binary image in which only the area of the label is in the foreground 
        #and the rest of the image is in the background   
        target = np.where(markers == label, 255, 0).astype(np.uint8)
       
        # Perform contour extraction on the created binary image
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        od.append(contours[0])
        
    
    output = cv2.drawContours(output, od, -1, color=(0, 23, 223), thickness=2) 
    ##center, radius = cv2.minEnclosingCircle(od[0])
   ## distances = [(cv2.pointPolygonTest(contours[0], center, True) )for point in contours[0]]
    distances = [np.sqrt((point[0][0] - center[0])**2 + (point[0][1] - center[1])**2) for point in contours[0]]
    radius =  sum(distances) / len(distances)
    diameter = int(2*radius)
   
    ##cv2.circle(output, (OD_center[1], OD_center[0]), int(radius), (255, 0, 0), 2)
    cv2.namedWindow('watershed', cv2.WINDOW_NORMAL)
    cv2.imshow('watershed', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output,diameter
    


def locate_point_P(od_center, centroid, diameter,image):
    output= cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Determine if the centroid is to the left or right of the OD center
    is_left = centroid[0] < od_center[1]
    distance_to_P = 2.5 * diameter
    # Draw a horizontal line from the OD center to the side where the centroid is
    if is_left:
        p = (od_center[1] - distance_to_P , od_center[0])
        
    else:
        p = (od_center[1] + distance_to_P, od_center[0])
       

    # Calculate the distance between OD center and point P
    distance_to_P = 2.5 * diameter
   

    # Draw the horizontal line and point P
    
    cv2.line(output, (0, int(od_center[0])), (int(output.shape[1]), int(od_center[0])), (127), 1)
    cv2.circle(output, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
    cv2.namedWindow('point p', cv2.WINDOW_NORMAL)
    cv2.imshow('point p', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return p

   
def extract_fovea_region(image, point, k):
    height, width= image.shape

    # Define a function to get a strip of width k pixels through point P
    def get_strip(image, point, k):
        x_center = point[0]

        # Extract the vertical strip centered at the given point
        x_start = int(max(x_center - k // 2, 0))
        x_end = int(min(x_center + k // 2, width))
        strip = image[:, x_start:x_end]
        return strip, x_start

    # Get the strip centered at the given point
    strip, x_start = get_strip(image, point, k)

    # Initialize variables to store maximum run length of zeros and its position
    max_run_length = 0
    start_pos = None
    end_pos = None

    # Slide the window along the strip
    chain_of_numbers = []
    # Slide the window starting from point P and going up
    for y in range(point[1], -1, -1):
         if y - k + 1 < 0:
             break
         window = strip[y-k+1:y+1, :]
         num_white_pixels = np.sum(window == 255)  # Counting white pixels (value 255)
         chain_of_numbers.append((y, num_white_pixels))
    
     # Slide the window starting from point P and going down
    for y in range(point[1] + 1, height - k + 1):
        window = strip[y:y+k, :]
        num_white_pixels = np.sum(window == 255)  # Counting white pixels (value 255)
        chain_of_numbers.append((y, num_white_pixels))

    # Find the maximum run length of zeros in the chain of numbers
    current_run_length = 0
    current_start = 0
    
    for i, (pos, count) in enumerate(chain_of_numbers):
        if count == 0:
            if current_run_length == 0:
                current_start = pos
            current_run_length += 1
            if current_run_length > max_run_length:
                max_run_length = current_run_length
                start_pos = current_start
                end_pos = pos
        else:
            current_run_length = 0

    if start_pos is not None and end_pos is not None:
        # Calculate the mid-position (D)
        D_y = (start_pos + end_pos) // 2
        D = (point[0], D_y)
        DS = abs(D_y - start_pos)
        # Check the calculated radius
        print(f"Start position: {start_pos}, End position: {end_pos}")
        print(f"Mid position (D): {D}, Radius (DS): {DS}")
        # Draw the circular region of interest
        image_with_circle = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.circle(image_with_circle, (int(point[0]),int( D_y)), DS, (0, 255, 0), 2)    
        cv2.namedWindow('fovea', cv2.WINDOW_NORMAL)
        cv2.imshow('fovea', image_with_circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return image_with_circle, D, DS
    else:
        return image, None, None


def extract_roi(gray_image, center, radius):
    # Create a binary image with a single black pixel at position D
    BW = np.zeros_like(gray_image, dtype=np.uint8)
    BW[center[1], int(center[0])] = 255  # Assuming center is in (y, x) format

    # Create a disc structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    # Dilate the binary image using the disc structuring element
    BW_dilated = cv2.dilate(BW, kernel)

    # Use the dilated binary image as a mask to extract pixels from the original grayscale image
    ROI = cv2.bitwise_and(gray_image, gray_image, mask=BW_dilated)
    
    cv2.namedWindow('Region of interest', cv2.WINDOW_NORMAL)
    cv2.imshow('Region of interest', ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ROI

def binarize_roi(roi):
    
    _, binarized_roi = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY)
    cv2.namedWindow('Binarize_roi', cv2.WINDOW_NORMAL)
    cv2.imshow('Binarize_roi', binarized_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return binarized_roi

def refine_macula_region(binarized_roi):
    # Perform morphological operations (erosion and dilation) to remove noise and smooth the macula region
    kernel = np.ones((5, 5), np.uint8)
    refined_roi = cv2.morphologyEx(binarized_roi, cv2.MORPH_OPEN, kernel)
    # Find contours in the refined ROI
    contours, _ = cv2.findContours(refined_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    macula_contour = None
    max_area = 2000
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            macula_contour = contour
        # Fit a circle to the contour
    ((center_x, center_y), radius) = cv2.minEnclosingCircle(macula_contour)
    center = (int(center_x), int(center_y))

       # Draw the fitted circle (blue color) on the original image
    image_with_circle = cv2.cvtColor(refined_roi, cv2.COLOR_GRAY2BGR)
    cv2.circle(image_with_circle, center, int(radius), (0, 255, 0), 2)
    cv2.circle(image_with_circle, center, 5, (0, 0, 255), 2)
    cv2.namedWindow('refined_roi_with_circle', cv2.WINDOW_NORMAL)
    cv2.imshow('refined_roi_with_circle', image_with_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image_with_circle, center , radius

def mask_outside_roi(original_image, mask):
   
    # Invert the mask (to mask the outside of the ROI)
    mask = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    cv2.namedWindow('refined_roi_with_circle', cv2.WINDOW_NORMAL)
    cv2.imshow('refined_roi_with_circle', masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return masked_image


def showCenters(image,fovea_center,fovea_radius,OD_center):
    output = image.copy()
    if fovea_center is not None:
        cv2.circle(output, fovea_center, int(fovea_radius), (0, 255, 0), 2)  # Fovea
        cv2.circle(output, fovea_center, 1, (0, 255,0 ), 5)
       
    if OD_center is not None:
        cv2.circle(output, (OD_center[1],OD_center[0]), 1, (255, 0, 0), 10)  # OD
    # Display the image
    cv2.namedWindow('Fovea and OD Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Fovea and OD Detection', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
# Example usage
image = cv2.imread("fundus_image.jpg")
gray_image = bgr2gray(image)
morphed_image = morph(gray_image)
binariezed_image = binariezed(morphed_image)
only_vessel_image = connectedComponentAnalysis(binariezed_image)
centroid = findCentroid(only_vessel_image)
temporal_removed_image = removeTemporal(centroid[0], only_vessel_image)
skeletonized = skeleton(temporal_removed_image)
pruned = prune(skeletonized)
OD_center = findIntersection(pruned)
image_od,diameter = watershed(image,OD_center)
point = locate_point_P(OD_center, centroid, diameter,only_vessel_image)
image_with_circle, D, DS = extract_fovea_region(only_vessel_image, point, 30)
region_of_interest=extract_roi(gray_image, D, DS)
binarized_roi = binarize_roi(region_of_interest)
masked_image = mask_outside_roi(region_of_interest, binarized_roi)
fovea_center_image , fovea_center,fovea_radius = refine_macula_region(masked_image)
showCenters(image, fovea_center, fovea_radius, OD_center)










