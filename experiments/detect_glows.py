import cv2 as cv
import numpy as np
from time import time
import detect_blade_utils as dbu

if __name__ == '__main__':
    def nothing(*arg):
        pass

palitra_win = 'palitra'
sett_def_win = 'settings_defect'
defect_win = 'Defect'

cv.namedWindow(palitra_win, cv.WINDOW_NORMAL)
cv.namedWindow(sett_def_win, cv.WINDOW_NORMAL)
cv.namedWindow(defect_win, cv.WINDOW_NORMAL)

cv.resizeWindow(palitra_win, 800, 300)
cv.resizeWindow(defect_win, 1000, 700)

cv.moveWindow(palitra_win, 0, 0)
cv.moveWindow(defect_win, 0, 400)
cv.moveWindow(sett_def_win, 1100, 0)

h1 = 16
s1 = 0
v1 = 0
h2 = 85
s2 = 255
v2 = 255
median_thresh = 57
gamma = 10
thresh_min = 45

# объявляем слайдеры для фильтров
cv.createTrackbar('Blur kernel', sett_def_win, blur_kernel, 20, nothing)
cv.createTrackbar('Gamma', sett_def_win, gamma, 30, nothing)
cv.createTrackbar('Green', sett_def_win, g, 1000, nothing)
cv.createTrackbar('Blue', sett_def_win, b, 1000, nothing)
cv.createTrackbar('Threshold min hsv', sett_def_win, thresh_min_hsv, 255, nothing)
cv.createTrackbar('Threshold max hsv', sett_def_win, thresh_max_hsv, 255, nothing)
# color settings
cv.createTrackbar('h1', sett_def_win, 33, 255, nothing)
cv.createTrackbar('s1', sett_def_win, 3, 255, nothing)
cv.createTrackbar('v1', sett_def_win, 0, 255, nothing)
cv.createTrackbar('h2', sett_def_win, 78, 255, nothing)
cv.createTrackbar('s2', sett_def_win, 255, 255, nothing)
cv.createTrackbar('v2', sett_def_win, 255, 255, nothing)

angle = 0
offset = 5
# img_path = '../../../Pictures/Baumer Image Records/VCXG-65C.R/20_uv.png'
img_path = '/home/vkorneychuk/projects/lum/photo/test1/{}.png'
max = 3
curr = max
img = cv.imread(img_path.format(curr), cv.IMREAD_COLOR)
img_palitra = cv.imread('./circle_rgb.jpg', cv.IMREAD_COLOR)
img_blank = np.zeros((img.shape[1], img.shape[0]), np.uint8)

while True:

    # считываем значения слайдеров
    kernel_side = cv.getTrackbarPos('Blur kernel', sett_def_win)
    gamma = cv.getTrackbarPos('Gamma', sett_def_win)
    g = cv.getTrackbarPos('Green', sett_def_win)
    b = cv.getTrackbarPos('Blue', sett_def_win)
    thresh_min_hsv = cv.getTrackbarPos('Threshold min hsv', sett_def_win)
    thresh_max_hsv = cv.getTrackbarPos('Threshold max hsv', sett_def_win)
    # считываем значения слайдеров hsv
    h1 = cv.getTrackbarPos('h1', sett_def_win)
    s1 = cv.getTrackbarPos('s1', sett_def_win)
    v1 = cv.getTrackbarPos('v1', sett_def_win)
    h2 = cv.getTrackbarPos('h2', sett_def_win)
    s2 = cv.getTrackbarPos('s2', sett_def_win)
    v2 = cv.getTrackbarPos('v2', sett_def_win)

    hsv_min = np.array((h1, s1, v1), np.uint8)
    hsv_max = np.array((h2, s2, v2), np.uint8)

    imgamma_palitra = dbu.apply_gamma(img_palitra, gamma)
    palitra_hsv = dbu.hsv_threshold(imgamma_palitra, imgamma_palitra, hsv_min, hsv_max)
    _, palitra_lights = dbu.denoise(palitra_hsv, imgamma_palitra, thresh_min)
    cv.imshow('palitra', palitra_lights)

    imgamma = dbu.apply_gamma(img, gamma)
    cv.imshow('test', imgamma)
    img_hsv_res_lights = dbu.hsv_threshold(imgamma, imgamma, hsv_min, hsv_max)
    orig_lights_wb = cv.cvtColor(img_hsv_res_lights, cv.COLOR_BGR2GRAY)
    thresh_img, denoise_blade_hsv = dbu.denoise(img_hsv_res_lights, imgamma, thresh_min)
    res = cv.addWeighted(img, 1.0, cv.cvtColor(thresh_img, cv.COLOR_GRAY2BGR), 0.5, 1)
    cv.imshow('test1', orig_lights_wb)
    cv.imshow('test3', thresh_img)

    pushed_key = cv.waitKey(1)
    # Enter
    if pushed_key == 13:
        pass
    elif pushed_key == 27:
        break
    elif pushed_key == 83:
        angle += offset
        if angle > 360:
            angle = 0
        img = cv.imread(img_path.format(angle), cv.IMREAD_COLOR)
    elif pushed_key == 81:
        angle -= offset
        if angle < 0:
            angle = 360
        img = cv.imread(img_path.format(angle), cv.IMREAD_COLOR)
    elif pushed_key == 83:
        curr = curr + 1 if curr < max else 0
        img = cv.imread(img_path.format(curr), cv.IMREAD_COLOR)
        img_gray = cv.imread(img_path.format(curr), cv.IMREAD_GRAYSCALE)
        print(curr)
    elif pushed_key == 81:
        curr = curr - 1 if curr > 0 else max
        img = cv.imread(img_path.format(curr), cv.IMREAD_COLOR)
        img_gray = cv.imread(img_path.format(curr), cv.IMREAD_GRAYSCALE)
        print(curr)
    elif pushed_key != -1:
        print(pushed_key)