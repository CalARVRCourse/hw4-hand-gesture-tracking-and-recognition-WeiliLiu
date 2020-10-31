import cv2
import numpy as np
import glob
import argparse
import logging

# Global Vars
window_name = "Hand Capture"
HSV_min_value = 0
SV_max_value = 255
H_max_value = 360
gaussian_blur_min = 1
gaussian_blur_max = 20
morpho_kernel_min = 1
morpho_kernel_max = 20
max_binary_value = 255

# This function records images from the connected camera to specified directory 
# when the "Space" key is pressed.
# directory: should be a string corresponding to the name of an existing 
# directory
def CaptureImages(directory, part=1, tune=False):
    # Open the camera for capture
    # the 0 value should default to the webcam, but you may need to change this
    # for your camera, especially if you are using a camera besides the default
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 turns off auto exp
    cam.set(cv2.CAP_PROP_AUTO_WB, 0.25) # 0.25 turns off auto wb
    cv2.namedWindow(window_name)

    # Create Trackbar to choose values for HSV lower bound
    cv2.createTrackbar('H_lo', window_name, 0, H_max_value, nothing)
    cv2.createTrackbar('S_lo', window_name, 0, SV_max_value, nothing)
    cv2.createTrackbar('V_lo', window_name, 0, SV_max_value, nothing)
    cv2.createTrackbar('H_hi', window_name, 0, H_max_value, nothing)
    cv2.createTrackbar('S_hi', window_name, 0, SV_max_value, nothing)
    cv2.createTrackbar('V_hi', window_name, 0, SV_max_value, nothing)
    # Create Trackbar to choose value for morphological transforms kernel
    # cv2.createTrackbar('kernel', window_name, morpho_kernel_min, morpho_kernel_max, nothing)
    # Create Trackbar to choose value for gaussian blur
    # cv2.createTrackbar('blur', window_name, gaussian_blur_min, gaussian_blur_max, nothing)
    # Create Trackbar to choose value for threshold
    cv2.createTrackbar('threshold', window_name, 0, 255, nothing)

    img_counter = 0
    # Read until user quits
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Extract current lower_HSV values
        H_lo, S_lo, V_lo = cv2.getTrackbarPos('H_lo', window_name), cv2.getTrackbarPos('S_lo', window_name), cv2.getTrackbarPos('V_lo', window_name)
        H_hi, S_hi, V_hi = cv2.getTrackbarPos('H_hi', window_name), cv2.getTrackbarPos('S_hi', window_name), cv2.getTrackbarPos('V_hi', window_name)

        lower_HSV = np.array([0, 44, 35], dtype="uint8")
        upper_HSV = np.array([241, 255, 36], dtype="uint8")

        convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)

        lower_YCrCb = np.array([0, 138, 67], dtype="uint8")
        upper_YCrCb = np.array([255, 173, 133], dtype="uint8")

        convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)

        skinMask = cv2.add(skinMaskHSV, skinMaskYCrCb)

        # skinMask = skinMaskHSV

        # Apply morphological transforms and standard gaussian blur
        # kernel_value = cv2.getTrackbarPos('kernel', window_name)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        blur_value = cv2.getTrackbarPos('blur', window_name)
        blur_value = blur_value + (blur_value % 2 == 0)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        frame = cv2.bitwise_and(frame, frame, mask=skinMask)

        # Binarize the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold_value = cv2.getTrackbarPos('threshold', window_name)

        if part == 1:
            ret, thresh = cv2.threshold(gray, 0, threshold_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # display the current image
            cv2.imshow("Display", thresh)

        if part == 2:
            ret, thresh = cv2.threshold(gray, 0, threshold_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # Connected components analysis
            ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh, ltype=cv2.CV_16U)
            markers = np.array(markers, dtype=np.uint8)
            label_hue = np.uint8(179 * markers / np.max(markers))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[label_hue==0] = 0

            if (ret > 2):
                try:
                    statsSortedByArea = stats[np.argsort(stats[:, 4])]
                    roi = statsSortedByArea[-3][0:4]
                    x, y, w, h = roi
                    subImg = labeled_img[y:y+h, x:x+w]
                    subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY)
                    _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    maxCntLength = 0
                    for i in range(0, len(contours)):
                        cntLength = len(contours[i])
                        if (cntLength > maxCntLength):
                            cnt = contours[i]
                            maxCntLength = cntLength
                        if (maxCntLength >= 5):
                            ellipseParam = cv2.fitEllipse(cnt)
                            subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB)
                            subImg = cv2.ellipse(subImg, ellipseParam, (0, 255, 0), 2)
                        
                        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                        print("(x, y) = ({}, {}), (MA, ma) = ({}, {}), angle = {}".format(x, y, MA, ma, angle))

                        subImg = cv2.resize(subImg, (0, 0), fx=3, fy=3)
                        cv2.imshow("ROI "+str(2), subImg)
                        cv2.waitKey(1)
                except:
                    print("No hand found")

            # display the current image
            cv2.imshow("Display", labeled_img)

        if part == 3:
            ret, thresh = cv2.threshold(gray, 0, threshold_value, cv2.THRESH_OTSU)
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            if len(contours) > 1:
                largestContour = contours[0]
                hull = cv2.convexHull(largestContour, returnPoints=False)
                for cnt in contours[:1]:
                    defects = cv2.convexityDefects(cnt, hull)
                    if (not isinstance(defects, type(None))):
                        for i in range(defects.shape[0]):
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(cnt[s][0])
                                end = tuple(cnt[e][0])
                                far = tuple(cnt[f][0])

                                cv2.line(thresh, start, end, [0, 255, 0], 2)
                                cv2.circle(thresh, far, 5, [0, 0, 255], -1)
                # Print center coordinates and the area of the contour
                M = cv2.moments(largestContour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                logging.info('Center: ({}, {}), Area: {}'.format(cX, cY, M))
            # display the current image
            cv2.imshow("Display", thresh)

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

def get_args():
    parser = argparse.ArgumentParser(description='CS294-137 HW4 Gesture Recognition')
    parser.add_argument('-p', '--part', metavar='E', type=int, default=1, help='Part of the homework to run', dest='part')
    parser.add_argument('-t', '--tune', metavar='T', type=str, nargs='?', const=True, default=False, help='Use the trackbar to tune parameters', dest='tune')
    return parser.parse_args()

# A function that does nothing
def nothing(x):
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print(args)
    CaptureImages("Hand-images", part=args.part)