import sys
import math
import cv2
import numpy as np

#================== Registration number ======================

# 1805841

#===================== Debug Mode ============================

DEBUG = False;

#================== Values in meters =========================

B = 3500

f = 12

pixel_spacing = 0.00001

#================ Colour Identifier List =====================

colours = ["White", "Red", "Green", "Blue", "Yellow", "Orange", "Cyan"]

# Boundries Colour order, Lower Upper
# The boundries were created based on the different values I got for each object when inspecting colour values from images

# White
# Red
# Green
# Blue
# Yellow
# Orange
# Cyan

colour_boundries = [
    ([99, 99, 99], [228, 228, 228]),
    ([0, 0, 89], [0, 0, 216]),
    ([0, 115, 0], [20, 218, 20]),
    ([97, 0, 0], [216, 0, 0]),
    ([0, 165, 165], [12, 217, 217]),
    ([0, 126, 172], [0, 154, 211]),
    ([170, 170, 0], [220, 220, 43])
    ]

#============== [frame, identifier, X, Y] ===================

# A empty 2D array which will store the frame number, identifier, and coordinates of each object for each frame

tracking_position_list = []

#=======================Information==========================

# The idea behind finding the UFO was that if I take away the coordinates of the previous position of an object
# from the new position i will get the difference in X and Y which i labeled as displacement in my code
# Once i calculate the displacement I can calculate the distance in pixel units that the object moved
# this can be done using C^2 = A^2 + B^2 where A and B can be either the X or Y displacement
# Once I have C I have the magnitude and if i divide the X and Y by the magnitude I will get a normalised displacement
# in the same direction, and now the magnitude will always be 1 for each normalised displacement

# Using this i decided to calculate an average normalised displacement and then compare each of the displacements to the average
# The way i can compare the displacements is by dividing a displacement by an average.
# By doing so I in theory should get values of around positive 1 because there might always be
# slight variation or error when using pixels as a way to getcoordinates.

# Now the way I was planning to identify the UFO would have been to have a margin of error that allowed the values to vary
# slightly from 1 and if they are beyond this error margin I would classify them as UFO.
# BUT the problem is that my contours and getting the coordinates from the images seem to have some outliers (possible colour boundires overlapping)
# and because of this I seem to get a very small amount of values that are not within the range and thus trigger the algorithm to mark the colour as a potential UFO


def main():
    print("frame   identity   distance")

    # Code to handle the invocation used by test harness
    # Code was provided in the Assignment PDF
    # Section 2 Operation and Evaluation
    frames = int (sys.argv[1])
    
    for frame in range (0, frames):
        fn_left = sys.argv[2] % frame
        fn_right = sys.argv[3] % frame

        im_left = cv2.imread (fn_left)
        im_right = cv2.imread (fn_right)

        im_leftY, im_leftX, im_leftC = im_left.shape
        im_rightY, im_rightX, im_rightC = im_right.shape

        colour_index = 0

        # Colour detection for the images was based on what I read in the link below 
        # https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
        
        for (lower, upper) in colour_boundries:
            lower = np.array (lower, dtype = "uint8")
            upper = np.array (upper, dtype = "uint8")

            contour_left = get_contours(im_left, lower, upper)
            contour_right = get_contours(im_right, lower, upper)

            if contour_left != None and contour_right != None:

                center_left = get_center (contour_left)
                center_right = get_center (contour_right)
            
                tracking_position_list.append ([frame, colour_index, center_left[0] - (im_leftX / 2), (im_leftY / 2) - center_left[1]])

                distance = get_coordinates (center_left[0], center_left[1], center_right[0], center_right[1], im_leftX, im_leftY, im_rightX, im_rightY)

                if DEBUG:
                    
                    print (frame, "     ", colours[colour_index], "     ", round(distance[2], 3), "     ", round(center_left[0] - (im_leftX / 2), 3), "     ", round((im_leftY / 2) - center_left[1], 3))
                                           
                else:
                    
                    print (frame, "     ", colours[colour_index], "     ", round(distance[2], 3))

            colour_index += 1

    find_UFO()

    
# Find contours for provided image using lower and upper boundries
# Part of Colour detection which was based on what i read in the link below but split into a method
# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
def get_contours(im, lower, upper):

    mask = cv2.inRange (im, lower, upper)
    
    output = cv2.bitwise_and (im, im, mask = mask)
    
    im_gray = cv2.cvtColor (output, cv2.COLOR_BGR2GRAY)
    
    contours, hierarchy = cv2.findContours (im_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len (contours) == 0:
        
        return None
    
    else:
        
        return contours

# Returns the centre position of the countoured object
# Due to the white dot in the middle of some of the objects it becomes black when masked and the program might return multiple countours for the same object
# this calculates the avg position by summing the coordinates and dividing the amount of contours found.
def get_center(contours):
    
    nContours = len (contours)

    sumCenterX = 0
    sumCenterY = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        sumCenterX += x
        sumCenterY += y

        XYList = [sumCenterX / nContours, sumCenterY / nContours]

    return XYList


# Calculate the distance of the object as well as a possible X based on Z solutions
def get_coordinates(xL, yL, xR, yR, im_leftX, im_leftY, im_rightX, im_rightY):
    
    # Z Solution from lecture
    xL -= (im_leftX / 2)
    xR -= (im_rightX / 2)

    xL *= pixel_spacing
    xR *= pixel_spacing
        
    Z = (f * B) / (xL - xR)

    # X Solution based on Z solution from lecture

    X = ((-1 * ((Z * xR) / f)) + (B - ((Z * xL) / f))) / 2

    # Put all coordinates into an array and return it

    XYZ = [X, 0, Z]

    return XYZ

# Find the UFO (Object not moving in a straight line) from the images (using only left side)
def find_UFO():

    # for division method
    error_margin = 1.1 
    
    # for angle method
    error_margin_ = 25 

    # Division method is a lot more sensitive to outliers
    USE_DIVISION_METHOD = False;

    list_UFO = []

    print_UFO = "UFO:"

    current_colour = 0

    temporary_tracking_list = []

    temporary_tracking_list_sig = []

    normalised_displacement_list = []

    avg_normalised_displacement = [0, 0]

    for c in colours:

        if DEBUG:
            print(c * 20)
            
        # Sort through all the recorded positions and based on the current colour move them in a new list to be processed 1 colour at a time
        for entry in tracking_position_list:

            if entry[1] == current_colour:

                temporary_tracking_list.append(entry)

        temporary_tracking_list_sig.append (temporary_tracking_list[0])
        

        # Sort through that list above but make sure that the minimum distance between 2 points is at least 2 pixel units to avoid inaccurate results affecting the algorithm
        for h in range (1, len (temporary_tracking_list) - 1):

            if 2 <= math.sqrt (((temporary_tracking_list[h][2] - temporary_tracking_list[h - 1][2]) ** 2) + ((temporary_tracking_list[h][3] - temporary_tracking_list[h - 1][3]) ** 2)):

                temporary_tracking_list_sig.append (temporary_tracking_list[h])
                                                    
        # Sort through the list above and calculate the displacement
        # Next calculate a normalised displacement and add the normalised displacement into a new list
        # Add the normalised displacement into a sum so that the displacements can be compared to a avg displacement
        # Make sure to wipe the lists from above after so that they are empty for the next colour 
        for i in range (1, len (temporary_tracking_list_sig) - 1):

            displacementX = temporary_tracking_list_sig[i][2] - temporary_tracking_list_sig[i - 1][2]
            displacementY = temporary_tracking_list_sig[i][3] - temporary_tracking_list_sig[i - 1][3]
    
            magnitude = math.sqrt ((displacementX ** 2) + (displacementY ** 2))

            if displacementX != 0 and displacementY != 0:

                normalised_displacement_list.append ([round(displacementX / magnitude, 3), round(displacementY / magnitude, 3)])

                if DEBUG:
                    print(round(displacementX / magnitude, 3), round(displacementY / magnitude, 3))

                avg_normalised_displacement[0] += round(displacementX / magnitude, 3)
                avg_normalised_displacement[1] += round(displacementY / magnitude, 3)

        temporary_tracking_list.clear()
        temporary_tracking_list_sig.clear()

        avg_normalised_displacement[0] /= len (normalised_displacement_list)
        avg_normalised_displacement[1] /= len (normalised_displacement_list)

        # Go through the list of displacements for a particular colour and divide them by the avg
        # Since the asteroids move in a straight line when we devide the displacement we should be getting a value of around 1
        # But if the direction changes we will get values that are much lower or higher than 1 and thus we identify the UFO
        # Once a UFO is identified we add the name of the colour (that was found to not move in a straight line) into a list of possible UFO's
        # The list will only keep 1 entry of a colour to avoid multiple identical colours being displayed
        if USE_DIVISION_METHOD:
            for d in normalised_displacement_list:

                displacement_differenceX =  avg_normalised_displacement[0] / d[0]
                displacement_differenceY = avg_normalised_displacement[1] / d[1]
                
                if DEBUG:
                    print ("DIFF | ", round(displacement_differenceX ,3), " | ", round(displacement_differenceY, 3), " |")

                if (1 - error_margin > displacement_differenceX) or (displacement_differenceX > 1 + error_margin) or (1 - error_margin > displacement_differenceY) or (displacement_differenceY > 1 + error_margin):
                    
                    list_UFO = add_UFO(list_UFO, colours[current_colour])

        # This is my second way of identifiying the UFO, instead of division I can use the normalised displacement vectors to calculate
        # the angle between the 2 vectors
        # This method seems more reliable as it is less sensitive to outliers in readings like the division method above
        # This also means I can use tighter margins of error compared to the method above
        else:
            for d in normalised_displacement_list:

                angle_variation = get_angle(avg_normalised_displacement, d)

                if angle_variation > error_margin_:

                    list_UFO = add_UFO(list_UFO, colours[current_colour])
                

        if DEBUG:
            print (" AVG N-DIS | ", avg_normalised_displacement[0], " | ",  avg_normalised_displacement[1], " |")
            print ("WIPE ============================================================")

        normalised_displacement_list.clear()

        avg_normalised_displacement = [0, 0]

        current_colour += 1
        
    # Go through the list of identified UFO's and add the colour identifier string into a string that is output at the end of the program so that the colour identifier is all in 1 line as requested
    for u in list_UFO:

        print_UFO = print_UFO + " "
        print_UFO = print_UFO + u
    
    print(print_UFO)
    

# Calculate the angle between 2 displacement vectors
# The maths for calculating the angle was based on a video
# https://www.youtube.com/watch?v=4hIh8ujylWE
def get_angle(avg_displacement, displacement):

    dot_product = (avg_displacement[0] * displacement[0]) + (avg_displacement[1] * displacement[1])

    angle = math.acos (dot_product)

    angle = np.rad2deg ( angle)

    if DEBUG:
        print ("Angle : ", angle)

    return angle

    
# Add the name of the colour which could be a possible UFO to a list (one of each colour only)
def add_UFO(list_UFO, Identifier):

    if len(list_UFO) == 0:
                                
        list_UFO.append (Identifier)
    
    else:

        already_exists = False

    for u in list_UFO:

        if u == Identifier:
                        
            already_exists = True

    if not already_exists:
                                
            list_UFO.append (Identifier)

    return list_UFO

main()

# Example of invocation by test harness
# python3 1805841.py 50 left-%3.3d.png right-%3.3d.png
