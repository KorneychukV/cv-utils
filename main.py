import sys
import cv2
import neoapi
import numpy as np
import time

hsv_from = np.array([100, 50, 50])
hsv_to = np.array([100, 50, 50])
new_exposure = 90000
current_exposure = new_exposure

def set_exposure(value):
    new_exposure = value
def set_h_from(value):
    hsv_from[0] = value
def set_s_from(value):
    hsv_from[1] = value
def set_v_from(value):
    hsv_from[2] = value

def set_h_to(value):
    hsv_to[0] = value
def set_s_to(value):
    hsv_to[1] = value
def set_v_to(value):
    hsv_to[2] = value

# get image and display (opencv)
result = 0
try:

    camera = neoapi.Cam()
    camera.Connect()

    camera.f.ExposureTime.Set(current_exposure)

    main_window = 'Press [ESC] to exit ..'
    cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
    exposure_window = 'Exposure window'
    cv2.namedWindow(exposure_window, cv2.WINDOW_NORMAL)
    hsv_window = 'HSV window'
    cv2.namedWindow(hsv_window, cv2.WINDOW_NORMAL)
    mask_window = 'Mask'
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    result_window = 'Result'
    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)

    cv2.createTrackbar('exposure', exposure_window,
                       int(camera.f.ExposureTime.GetMin()), int(camera.f.ExposureTime.GetMax()),
                       set_exposure)
    cv2.setTrackbarPos('exposure', exposure_window, current_exposure)


    cv2.createTrackbar('Hue From', hsv_window, 0, 255, set_h_from)
    cv2.setTrackbarPos('Hue From', hsv_window, hsv_from[0])
    cv2.createTrackbar('Saturation From', hsv_window, 0, 255, set_s_from)
    cv2.setTrackbarPos('Saturation From', hsv_window, hsv_from[1])
    cv2.createTrackbar('Value From', hsv_window, 0, 255, set_v_from)
    cv2.setTrackbarPos('Value From', hsv_window, hsv_from[2])

    cv2.createTrackbar('Hue To', hsv_window, 0, 255, set_h_to)
    cv2.setTrackbarPos('Hue To', hsv_window, hsv_to[0])
    cv2.createTrackbar('Saturation To', hsv_window, 0, 255, set_s_to)
    cv2.setTrackbarPos('Saturation To', hsv_window, hsv_to[1])
    cv2.createTrackbar('Value To', hsv_window, 0, 255, set_v_to)
    cv2.setTrackbarPos('Value To', hsv_window, hsv_to[2])

    while True:
        new_exposure = cv2.getTrackbarPos('exposure', exposure_window)
        if new_exposure != current_exposure:
            current_exposure = new_exposure
            camera.f.ExposureTime.Set(current_exposure)
        img = camera.GetImage().GetNPArray()
        cv2.imshow(main_window, img)

        mask_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(mask_img, hsv_from, hsv_to)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow(mask_window, mask)
        cv2.imshow(result_window, res)

        pushed_key = cv2.waitKey(1)
        if pushed_key == 13:
            cv2.imwrite('opencv_python.bmp', img)
        if pushed_key == 27:
            break

except (neoapi.NeoException, Exception) as exc:
    print('error: ', exc.GetDescription())
    result = 1


sys.exit(result)
