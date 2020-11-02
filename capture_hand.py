import argparse
import glob
import logging
import cv2
import numpy as np
import pyautogui

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
# Camera Resolutions
screen_width_res = 3840
screen_height_res = 2160
cam_width_res = 1280
cam_height_res = 720
# pyautogui constants
offsetX = 0
offsetY = 0
scaleX = screen_height_res / cam_height_res
scaleY = screen_width_res / cam_width_res

# This function records images from the connected camera to specified directory 
# when the "Space" key is pressed.
# directory: should be a string corresponding to the name of an existing 
# directory
def CaptureImages(directory, part=1, tune=False, heuristic=False, twa=False):
    # Open the camera for capture
    # the 0 value should default to the webcam, but you may need to change this
    # for your camera, especially if you are using a camera besides the default
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 turns off auto exp
    cam.set(cv2.CAP_PROP_AUTO_WB, 0.25) # 0.25 turns off auto wb
    # Setting camera resolutions
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width_res)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height_res)
    if tune:
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

    # Variables for decreasing opencv noise
    r_vals = []
    g_vals = []
    b_vals = []
    (rAvg, gAvg, bAvg) = (None, None, None)
    total = 0
    angleAvg = None

    img_counter = 0
    prevArea = 0
    prevAngle = 0
    prevCX = 0
    prevCY = 0
    # Read until user quits
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        if twa:
            (B, G, R) = cv2.split(frame.astype("float"))
            if rAvg is None or total == 3:
                rAvg, gAvg, bAvg = R, G, B
                total = 0
            else:
                rAvg = ((total * rAvg) + (1 * R)) / (total + 1.0)
                gAvg = ((total * gAvg) + (1 * G)) / (total + 1.0)
                bAvg = ((total * bAvg) + (1 * B)) / (total + 1.0)
            total += 1
            frame = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")

        """
        Showing only the unprocessed camera frame
        """
        if part == 0:
            # display the current image
            cv2.imshow("Display", frame)

            # wait for 1ms or key press
            k = cv2.waitKey(1) #k is the key pressed
            if k == 27 or k == 113: #27, 113 are ascii for escape and q respectively
                #exit
                break
            continue

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

        # attempting to add a medianblur filter to reduce salt/pepper pixel effects
        frame = cv2.medianBlur(frame, 5)
        # and then apply a bilateral filter so the edges are not affected by noise reduction
        frame = cv2.bilateralFilter(frame, 9, 75, 75) #diameter, sigmaColor for color space, sigmaColor for coordinatespace. larger value for sigma color means colors far apart will mix, and more pixels too.

        if part == 1:
            cv2.imshow("Display", frame)

        # Binarize the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold_value = cv2.getTrackbarPos('threshold', window_name)

        if part == 2:
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
                    # _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    maxCntLength = 0
                    for i in range(0, len(contours)):
                        cntLength = len(contours[i])
                        if (cntLength > maxCntLength):
                            cnt = contours[i]
                            maxCntLength = cntLength
                    if (maxCntLength >= 5):
                        ellipseParam = cv2.fitEllipse(cnt)
                        (x, y), (MA, ma), angle = ellipseParam
                        print("(x, y) = ({}, {}), (MA, ma) = ({}, {}), angle = {}".format(x, y, MA, ma, angle))
                        text = "(x, y) = ({}, {})".format(x, y)
                        cv2.putText(labeled_img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                        text = "(MA, ma) = ({}, {})".format(MA, ma)
                        cv2.putText(labeled_img, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                        text = "angle = {}".format(angle)
                        cv2.putText(labeled_img, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                        # if part == 4:
                        #     cX = offsetX + scaleX * x
                        #     cY = offsetY + scaleY * y
                        #     print(cX, cY)
                        #     pyautogui.moveTo(cX, cY, duration=0.02, tween=pyautogui.easeInOutQuad)
                        subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB)
                        subImg = cv2.ellipse(subImg, ellipseParam, (0, 255, 0), 2)

                    subImg = cv2.resize(subImg, (0, 0), fx=3, fy=3)
                    cv2.imshow("ROI "+str(2), subImg)
                    cv2.waitKey(1)
                except:
                    print("No hand found")

            # display the current image
            cv2.imshow("Display", labeled_img)

        if part == 3 or part == 4:
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            fingerAngles = []
            if len(contours) > 1:
                largestContour = contours[0]
                hull = cv2.convexHull(largestContour, returnPoints=False)
                for cnt in contours[:1]:
                    defects = cv2.convexityDefects(cnt, hull)
                    if (not isinstance(defects, type(None))):
                        fingerCount = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])

                            if heuristic:
                                # Defect check
                                c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                                a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2
                                b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2
                                angle = np.arccos((a_squared + b_squared - c_squared) / (2 * np.sqrt(a_squared * b_squared)))

                                if angle <= np.pi / 3:
                                    fingerAngles.append(angle)
                                    fingerCount += 1
                                    cv2.line(thresh, start, end, [0, 255, 0], 2)
                                    cv2.circle(thresh, far, 4, [0, 0, 255], -1)
                            else:
                                cv2.line(thresh, start, end, [0, 255, 0], 2)
                                cv2.circle(thresh, far, 4, [0, 0, 255], -1)

                        fingerCount = fingerCount + int(fingerCount != 0)
                        if part == 3:
                            text = "Finger count: " + str(fingerCount)
                            cv2.putText(thresh, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                # Print center coordinates and the area of the contour
                if part == 4:
                    M = cv2.moments(largestContour)
                    area = cv2.contourArea(largestContour)
                    cX = offsetX + scaleX * int(M["m10"] / M["m00"])
                    cY = offsetY + scaleY * int(M["m01"] / M["m00"])
                    (_, _), (_, _), angle = cv2.fitEllipse(largestContour)
                    text = "angle: " + str(angle)
                    cv2.putText(thresh, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                    pyautogui.moveTo(cX, cY, duration=0.02, tween=pyautogui.easeInOutQuad)
                    if fingerCount == 2:
                        pyautogui.rightClick()
                    if fingerCount == 3:
                        pyautogui.click()
                    if fingerCount == 0:
                        pyautogui.press('esc')
                    if isIncreased(area, prevArea, 40000) and np.abs(prevCX - cX) < 200:
                        ZoomIn()
                    elif isDecreased(area, prevArea, 40000) and np.abs(prevCX - cX) < 200:
                        ZoomOut()

                    if isIncreased(angle, prevAngle, 35) and np.abs(prevCX - cX) < 200 and angle < 100:
                        print("Turn Right")
                        RotateLeft()
                    # elif isDecreased(angle, prevAngle, 35) and np.abs(prevCX - cX) < 200:
                    #     print("Turn Left")
                    #     RotateRight()
                    print(cX - prevCX)
                    if isIncreased(cX, prevCX, 800):
                        print("next photo")
                        nextPhoto()
                    elif isDecreased(cX, prevCX, 800):
                        prevPhoto()

                    if isIncreased(cY, prevCY, 2000):
                        pyautogui.press('esc')
                    
                    prevArea = area
                    prevAngle = angle
                    prevCX = cX
                    prevCY = cY
                # logging.info('Center: ({}, {}), Area: {}'.format(cX, cY, M))
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

def isIncreased(area, prevArea, threshold):
    if area > prevArea + threshold:
        return True
    else:
        return False

def isDecreased(area, prevArea, threshold):
    if area < prevArea - threshold:
        return True
    else:
        return False

def ZoomIn():
    pyautogui.hotkey('ctrl', '+')

def ZoomOut():
    pyautogui.hotkey('ctrl', '-')

def RotateRight():
    pyautogui.hotkey('ctrl', 'r')

def RotateLeft():
    pyautogui.hotkey('ctrl', 'r', presses=3)

def nextPhoto():
    pyautogui.scroll(-1)

def prevPhoto():
    pyautogui.scroll(1)

def get_args():
    parser = argparse.ArgumentParser(description='CS294-137 HW4 Gesture Recognition')
    parser.add_argument('-p', '--part', metavar='E', type=int, default=1, help='Part of the homework to run', dest='part')
    parser.add_argument('-t', '--tune', metavar='T', type=str, nargs='?', const=True, default=False, help='Use the trackbar to tune parameters', dest='tune')
    parser.add_argument('-hf', '--heuristic-filtering', metavar='H', type=str, nargs='?', const=True, default=False, help='Apply heuristic filtering to defects', dest='heu_filtering')
    parser.add_argument('-twa', '--time-window-averaging', metavar='A', type=str, nargs='?', const=True, default=False, help='Apply time window averaging', dest='time_window_averaging')
    return parser.parse_args()

# A function that does nothing
def nothing(x):
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print(args)
    CaptureImages("Hand-images", part=args.part, tune=args.tune, heuristic=args.heu_filtering, twa=args.time_window_averaging)
