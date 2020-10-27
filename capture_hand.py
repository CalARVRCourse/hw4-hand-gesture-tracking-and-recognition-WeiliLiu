import cv2
import numpy as np
import glob

# Global Vars
window_name = "Hand Capture"
HSV_min_value = 0
SV_max_value = 255
H_max_value = 25
gaussian_blur_min = 1
gaussian_blur_max = 20
morpho_kernel_min = 1
morpho_kernel_max = 20

# This function records images from the connected camera to specified directory 
# when the "Space" key is pressed.
# directory: should be a string corresponding to the name of an existing 
# directory
def CaptureImages(directory):
    # Open the camera for capture
    # the 0 value should default to the webcam, but you may need to change this
    # for your camera, especially if you are using a camera besides the default
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)

    # Create Trackbar to choose values for HSV lower bound
    cv2.createTrackbar('H', window_name, HSV_min_value, H_max_value, nothing)
    cv2.createTrackbar('S', window_name, HSV_min_value, SV_max_value, nothing)
    cv2.createTrackbar('V', window_name, HSV_min_value, SV_max_value, nothing)
    # Create Trackbar to choose value for morphological transforms kernel
    cv2.createTrackbar('kernel', window_name, morpho_kernel_min, morpho_kernel_max, nothing)
    # Create Trackbar to choose value for gaussian blur
    cv2.createTrackbar('blur', window_name, gaussian_blur_min, gaussian_blur_max, nothing)

    img_counter = 0
    # Read until user quits
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Extract current lower_HSV values
        H_lo, S_lo, V_lo = cv2.getTrackbarPos('H', window_name), cv2.getTrackbarPos('S', window_name), cv2.getTrackbarPos('V', window_name)

        lower_HSV = np.array([60, 100, 100], dtype="uint8")
        upper_HSV = np.array([25, 255, 255], dtype="uint8")

        convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)

        lower_YCrCb = np.array([0, 138, 67], dtype="uint8")
        upper_YCrCb = np.array([255, 173, 133], dtype="uint8")

        convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)

        skinMask = cv2.add(skinMaskHSV, skinMaskYCrCb)

        # # Apply morphological transforms and standard gaussian blur
        # kernel_value = cv2.getTrackbarPos('kernel', window_name)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # # blur the mask to help remove noise, then apply the
        # # mask to the frame
        # blur_value = cv2.getTrackbarPos('blur', window_name)
        # blur_value = blur_value + (blur_value % 2 == 0)
        # skinMask = cv2.GaussianBlur(skinMask, (blur_value, blur_value), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        # display the current image
        cv2.imshow("Display", skin)
        # wait for 1ms or key press
        k = cv2.waitKey(1) #k is the key pressed
        if k == 27 or k == 113: #27, 113 are ascii for escape and q respectively
            #exit
            break
        elif k == 32: #32 is ascii for space
            #record image
            img_name = "hand_image_{}.png".format(img_counter)
            cv2.imwrite(directory+'/'+img_name, frame)
            print("Writing: {}".format(directory+'/'+img_name))
            img_counter += 1
    cam.release()

# A function that does nothing
def nothing(x):
    pass

if __name__ == "__main__":
    CaptureImages("Hand-images")