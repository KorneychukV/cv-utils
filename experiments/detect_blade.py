import cv2 as cv
import numpy as np
from time import time
import detect_blade_utils as dbu

if __name__ == '__main__':
    def nothing(*arg):
        pass

palitra_win = 'palitra'
sett_blade_win = 'settings_blade'
blade_win = 'Blade'

cv.namedWindow(palitra_win, cv.WINDOW_NORMAL)
cv.namedWindow(sett_blade_win, cv.WINDOW_NORMAL)
cv.namedWindow(blade_win, cv.WINDOW_NORMAL)
cv.resizeWindow(blade_win, 1000, 700)

blur_kernel = 7
b = 394
g = 453
r = 871
thresh = 10
br = 255
contr = 127

# создаем слайдеры для настройки фильтра
cv.createTrackbar('Blur kernel', sett_blade_win, blur_kernel, 20, nothing)
cv.createTrackbar('Red', sett_blade_win, r, 1000, nothing)
cv.createTrackbar('Green', sett_blade_win, g, 1000, nothing)
cv.createTrackbar('Blue', sett_blade_win, b, 1000, nothing)
cv.createTrackbar('Threshold min', sett_blade_win, thresh, 100, nothing)
cv.createTrackbar('Sobel', sett_blade_win, thresh, 15, nothing)
cv.createTrackbar('Bright', sett_blade_win, br, 510, nothing)
cv.createTrackbar('Contrast', sett_blade_win, contr, 254, nothing)

offset = 5
# img_path = '../../lum/automated-fpi-line/modbus-connector/test_photo/blade/cam_1_lum_reflector/{}.png'
# img_path = 'uv.png'
img_path = '../../lum/photo/19.10/7/uv-type_2-2-1-{}.png'
angles = ['30', '120', '210', '300']
angle = 0

img = cv.imread(img_path.format(angles[angle]), cv.IMREAD_COLOR)
img_gray = cv.imread(img_path.format(angles[angle]), cv.IMREAD_GRAYSCALE)
img_palitra = cv.imread('./circle_rgb.jpg', cv.IMREAD_COLOR)

blue_coef = -1.06
green_coef = -0.47
red_coef = 3.71

img_blank = np.zeros((img.shape[1], img.shape[0]), np.uint8)

while True:

    # считываем значения слайдеров
    kernel_side = cv.getTrackbarPos('Blur kernel', sett_blade_win)
    r = cv.getTrackbarPos('Red', sett_blade_win)
    g = cv.getTrackbarPos('Green', sett_blade_win)
    b = cv.getTrackbarPos('Blue', sett_blade_win)
    thresh_min = cv.getTrackbarPos('Threshold min', sett_blade_win)
    sobel = cv.getTrackbarPos('Sobel', sett_blade_win) * 2 + 1
    br = cv.getTrackbarPos('Bright', sett_blade_win)
    contr = cv.getTrackbarPos('Contrast', sett_blade_win)

    # # Color IMAGE
    # # Show palitra
    # palitra_res = dbu.mono_mixer(img_palitra, b, g, r)
    # cv.imshow(palitra_win, palitra_res)
    # # Apply blur
    # blur_img = dbu.blur_img(img, kernel_side)
    # mix_img = dbu.mono_mixer(blur_img, b, g, r)
    # thresh_img, denoise_blade = dbu.denoise(mix_img, img, thresh_min)
    # cv.imshow(blade_win, dbu.stackImages([[img, blur_img],
    #                                       [thresh_img, denoise_blade]], 1))

    # GRAYSCALE IMAGE
    img_bc = dbu.bright_contr(img, br, contr)
    blur_img = dbu.blur_img(img_bc, kernel_side)
    img_gray = cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY)
    _, thresh_gray = cv.threshold(img_gray, thresh_min, 255, cv.THRESH_BINARY)
    # blade_shape = cv.bitwise_and(img_gray, img_gray, mask=thresh_gray)
    cv.imshow(blade_win, dbu.stackImages([[blur_img, thresh_gray]], 1))

    pushed_key = cv.waitKey(1)

    # Enter
    if pushed_key == 13:
        pass
    elif pushed_key == 27:
        break
    elif pushed_key == 83:
        angle += 1
        if angle > 3:
            angle = 0
        img = cv.imread(img_path.format(angles[angle]), cv.IMREAD_COLOR)
    elif pushed_key == 81:
        angle -= offset
        if angle < 0:
            angle = 3
        img = cv.imread(img_path.format(angles[angle]), cv.IMREAD_COLOR)
    elif pushed_key != -1:
        print(pushed_key)