import numpy as np
import cv2
import serial
import time

cap = cv2.VideoCapture(2)
img_counter = 0
stage = 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

main_window = 'Press [ESC] to exit ..'
cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    cv2.imshow(main_window, frame)

    pushed_key = cv2.waitKey(1)
    if pushed_key == 13:
        cv2.imwrite("img{}.png".format(img_counter), frame)
        img_counter += 1
    elif pushed_key == 27:
        break