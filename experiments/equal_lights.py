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

blur_kernel = 9
b = 157
g = 1000
r = 269
thresh_max_hsv = 255
thresh_min_hsv = 39

# объявляем слайдеры для фильтров
cv.createTrackbar('Blur kernel', sett_def_win, blur_kernel, 20, nothing)
cv.createTrackbar('Red', sett_def_win, r, 1000, nothing)
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
img_path = '../../lum/photo/06.10/7/uv-type_2-1-1-120.png'
img = cv.imread(img_path.format(angle), cv.IMREAD_COLOR)
img_palitra = cv.imread('./circle_rgb.jpg', cv.IMREAD_COLOR)
img_blank = np.zeros((img.shape[1], img.shape[0]), np.uint8)

while True:

    # считываем значения слайдеров
    kernel_side = cv.getTrackbarPos('Blur kernel', sett_def_win)
    r = cv.getTrackbarPos('Red', sett_def_win)
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

    # Apply hsv threshold for palitra
    hsv_min = np.array((h1, s1, v1), np.uint8)
    hsv_max = np.array((h2, s2, v2), np.uint8)
    palitra_hsv_res = dbu.hsv_threshold(img_palitra, img_palitra, hsv_min, hsv_max)
    # Apply Mono mixer for palitra
    palitra_mix_res = dbu.mono_mixer(img_palitra, b, g, r)
    cv.imshow(palitra_win, dbu.stackImages([[palitra_mix_res, palitra_hsv_res]], 1))

    # Apply
    blur_img = dbu.blur_img(img, kernel_side)
    # mix_img = dbu.mono_mixer(blur_img, b, g, r)
    img_hsv_res = dbu.hsv_threshold(blur_img, blur_img, hsv_min, hsv_max)
    # thresh_img_mix, denoise_blade_mix = dbu.denoise(mix_img, img, thresh_min_mix)
    thresh_img_hsv, denoise_blade_hsv = dbu.denoise(img_hsv_res, img, thresh_min_hsv)

    cv.imshow(defect_win, dbu.stackImages([[img, blur_img],
                                            [thresh_img_hsv, denoise_blade_hsv]],1))
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
    elif pushed_key != -1:
        print(pushed_key)