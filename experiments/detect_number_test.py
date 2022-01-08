import cv2 as cv
import numpy as np
from time import time
import detect_blade_utils as dbu
import easyocr
import cv2

def recognize(img, reader):
    img_r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    horizontal_list, free_list = reader.detect(img_r)
    result = reader.recognize(img_r, horizontal_list=horizontal_list, free_list=free_list)
    print(result)
    return


def __load_settings(settings):
    angle = settings[0]
    t_in = settings[1:9].reshape(-1, 2).astype('float32')
    t_out = settings[9:17].reshape(-1, 2).astype('float32')
    return angle, t_in, t_out


def __transform_image(img, rotate_angle, t_in, t_out):
    h, w = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), rotate_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    trans = cv2.getPerspectiveTransform(t_in, t_out)
    transformed = cv2.warpPerspective(rotated, trans, (w, h), flags=cv2.INTER_LINEAR)
    return transformed


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

blur_kernel = 4
alpha = 300
alpha_1 = 106
beta = 237
beta_1 = 96
erode = 0
dilate = 0
thresh_max_hsv = 255
thresh_min_hsv = 39
morphs = [cv.MORPH_CLOSE, cv.MORPH_OPEN, cv.MORPH_HITMISS, cv.MORPH_BLACKHAT, cv.MORPH_TOPHAT]
shapes = [cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS]

# объявляем слайдеры для фильтров
cv.createTrackbar('Blur kernel', sett_def_win, blur_kernel, 20, nothing)
cv.createTrackbar('Bright', sett_def_win, alpha, 510, nothing)
cv.createTrackbar('Contr', sett_def_win, beta, 254, nothing)
cv.createTrackbar('Bright_1', sett_def_win, alpha_1, 510, nothing)
cv.createTrackbar('Contr_1', sett_def_win, beta_1, 254, nothing)
cv.createTrackbar('Erode', sett_def_win, 0, 10, nothing)
cv.createTrackbar('Dilate', sett_def_win, 0, 10, nothing)
cv.createTrackbar('Morph_kernel', sett_def_win, 0, 60, nothing)
cv.createTrackbar('Morph_shape', sett_def_win, 0, len(shapes)-1, nothing)
cv.createTrackbar('Morph_index', sett_def_win, 0, len(morphs)-1, nothing)
cv.createTrackbar('Threshold min hsv', sett_def_win, thresh_min_hsv, 255, nothing)
cv.createTrackbar('Threshold max hsv', sett_def_win, thresh_max_hsv, 255, nothing)
# color settings
cv.createTrackbar('h1', sett_def_win, 53, 255, nothing)
cv.createTrackbar('s1', sett_def_win, 40, 255, nothing)
cv.createTrackbar('v1', sett_def_win, 0, 255, nothing)
cv.createTrackbar('h2', sett_def_win, 78, 255, nothing)
cv.createTrackbar('s2', sett_def_win, 255, 255, nothing)
cv.createTrackbar('v2', sett_def_win, 255, 255, nothing)

angle = 0
offset = 5
img_path = '/var/lib/docker/volumes/docker_media/_data/LEAP-1B STEP 2/710/snapshot_number.png'
img = cv.imread(img_path.format(angle), cv.IMREAD_GRAYSCALE)

# rot_angle, t_in, t_out = __load_settings(np.load('settings.npy'))
# img = __transform_image(img, rot_angle, t_in, t_out)
print(img.shape)

img = img[450:750, 1200:1500]
img = np.rot90(img, 2)
img_palitra = cv.imread('./circle_rgb.jpg', cv.IMREAD_COLOR)
img_blank = np.zeros((img.shape[1], img.shape[0]), np.uint8)

reader = easyocr.Reader(['en'], gpu=True)

while True:

    # считываем значения слайдеров
    kernel_side = cv.getTrackbarPos('Blur kernel', sett_def_win)
    kernel_side = kernel_side if kernel_side > 0 else 1
    alpha = cv.getTrackbarPos('Bright', sett_def_win)
    alpha_1 = cv.getTrackbarPos('Bright_1', sett_def_win)
    beta = cv.getTrackbarPos('Contr', sett_def_win)
    beta_1 = cv.getTrackbarPos('Contr_1', sett_def_win)
    erode = cv.getTrackbarPos('Erode', sett_def_win)
    dilate = cv.getTrackbarPos('Dilate', sett_def_win)
    morph_index = cv.getTrackbarPos('Morph_index', sett_def_win)
    shape_index = cv.getTrackbarPos('Morph_shape', sett_def_win)
    morph_kernel = cv.getTrackbarPos('Morph_kernel', sett_def_win)

    thresh_min_hsv = cv.getTrackbarPos('Threshold min hsv', sett_def_win)
    thresh_max_hsv = cv.getTrackbarPos('Threshold max hsv', sett_def_win)
    # считываем значения слайдеров hsv
    h1 = cv.getTrackbarPos('h1', sett_def_win)
    s1 = cv.getTrackbarPos('s1', sett_def_win)
    v1 = cv.getTrackbarPos('v1', sett_def_win)
    h2 = cv.getTrackbarPos('h2', sett_def_win)
    s2 = cv.getTrackbarPos('s2', sett_def_win)
    v2 = cv.getTrackbarPos('v2', sett_def_win)
    hsv_min = (h1, s1, v1)
    hsv_max = (h2, s2, v2)

    img_hsv = dbu.hsv_threshold(img, img, hsv_min, hsv_max)
    _, img_mask = cv.threshold(cv.cvtColor(img_hsv, cv.COLOR_BGR2GRAY), thresh_min_hsv, thresh_max_hsv, cv.THRESH_BINARY)
    _, img_mask_inv = cv.threshold(cv.cvtColor(img_hsv, cv.COLOR_BGR2GRAY), thresh_min_hsv, thresh_max_hsv, cv.THRESH_BINARY_INV)
    img_com = cv.bitwise_and(img, img, mask=img_mask_inv)

    img_ligh = cv.bitwise_and(img, img, mask=img_mask)
    img_bc = dbu.bright_contr(img_ligh, alpha, beta)
    img_bc_1 = dbu.bright_contr(img_bc, alpha_1, beta_1)
    # img_ligh_inv = np.invert(img_bc)
    img_ligh_res = cv.bitwise_and(img_bc_1, img_bc_1, mask=img_mask)
    img_res = img_ligh_res+img_com
    img_res = cv.cvtColor(img_res, cv.COLOR_BGR2GRAY)

    img_blur = cv.blur(img_res, (kernel_side,kernel_side))

    # img_gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
    img_gray = img_blur
    gradX = cv.Sobel(img_gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv.Sobel(img_gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=-1)
    # # subtract the y-gradient from the x-gradient
    gradient = cv.subtract(gradX, gradY)
    gradient = cv.convertScaleAbs(gradient)

    img_grad_bc = dbu.bright_contr(gradient, alpha_1, beta_1)

    morph_kernel = 1 if morph_kernel <= 0 else morph_kernel
    kernel = cv.getStructuringElement(shapes[shape_index], (morph_kernel, morph_kernel))
    closed = cv.morphologyEx(img_grad_bc, morphs[morph_index], kernel)
    # closed = cv.erode(img_thresh, None, iterations=erode)
    # closed = cv.dilate(closed, None, iterations=dilate)

    gradX = cv.Sobel(closed, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv.Sobel(closed, ddepth=cv.CV_32F, dx=0, dy=1, ksize=-1)
    # # subtract the y-gradient from the x-gradient
    gradient = cv.subtract(gradX, gradY)
    gradient = cv.convertScaleAbs(gradient)


    cv.imshow(defect_win, dbu.stackImages([[img, img_res, img_blur], [gradient, img_grad_bc, closed]],1))
    pushed_key = cv.waitKey(1)
    # Enter
    if pushed_key == 13:
        print(1)
        recognize(img_res, reader)
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