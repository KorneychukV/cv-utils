import cv2 as cv
import numpy as np
from time import time
import detect_blade_utils as dbu

def apply_gamma(img, gamma):
    gamma = gamma if gamma != 0 else 0.01
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(img, table)

def fourie(I):
    rows, cols = I.shape
    m = cv.getOptimalDFTSize(rows)
    n = cv.getOptimalDFTSize(cols)
    padded = cv.copyMakeBorder(I, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])

    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv.merge(planes)  # Add to the expanded another plane with zeros

    cv.dft(complexI, complexI)  # this way the result may fit in the source matrix

    cv.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv.magnitude(planes[0], planes[1], planes[0])  # planes[0] = magnitude
    magI = planes[0]

    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv.add(matOfOnes, magI, magI)  # switch to logarithmic scale
    cv.log(magI, magI)

    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows / 2)
    cy = int(magI_cols / 2)
    q0 = magI[0:cx, 0:cy]  # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx + cx, 0:cy]  # Top-Right
    q2 = magI[0:cx, cy:cy + cy]  # Bottom-Left
    q3 = magI[cx:cx + cx, cy:cy + cy]  # Bottom-Right
    tmp = np.copy(q0)  # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)  # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp

    cv.normalize(magI, magI, 0, 1, cv.NORM_MINMAX)
    return magI

if __name__ == '__main__':
    def nothing(*arg):
        pass

palitra_win = 'palitra'
sett_def_win = 'settings_defect'
defect_win = 'Defect'

# cv.namedWindow(palitra_win, cv.WINDOW_NORMAL)
cv.namedWindow(sett_def_win, cv.WINDOW_NORMAL)
cv.namedWindow(defect_win, cv.WINDOW_NORMAL)
cv.namedWindow('test', cv.WINDOW_NORMAL)

cv.resizeWindow(palitra_win, 800, 300)
cv.resizeWindow(defect_win, 1000, 700)

cv.moveWindow(palitra_win, 0, 0)
cv.moveWindow(defect_win, 0, 400)
cv.moveWindow(sett_def_win, 1100, 0)

blur_kernel = 5
b = 157
g = 1000
dk = 20
thresh_max_hsv = 255
thresh_min_hsv = 55
gamma = 146

# объявляем слайдеры для фильтров
cv.createTrackbar('Blur kernel', sett_def_win, blur_kernel, 20, nothing)
cv.createTrackbar('Dilate kernel', sett_def_win, dk, 20, nothing)
cv.createTrackbar('Green', sett_def_win, g, 1000, nothing)
cv.createTrackbar('Blue', sett_def_win, b, 1000, nothing)
cv.createTrackbar('Threshold min hsv', sett_def_win, thresh_min_hsv, 255, nothing)
cv.createTrackbar('Threshold max hsv', sett_def_win, thresh_max_hsv, 255, nothing)
cv.createTrackbar('Gamma', sett_def_win, gamma, 400, nothing)
# color settings
cv.createTrackbar('h1', sett_def_win, 16, 255, nothing)
cv.createTrackbar('s1', sett_def_win, 0, 255, nothing)
cv.createTrackbar('v1', sett_def_win, 0, 255, nothing)
cv.createTrackbar('h2', sett_def_win, 78, 255, nothing)
cv.createTrackbar('s2', sett_def_win, 255, 255, nothing)
cv.createTrackbar('v2', sett_def_win, 255, 255, nothing)

offset = 5
inspection_id = 1
# img_path = '/var/lib/docker/volumes/docker_media/_data/LEAP-1B STEP 2/{}/{}'
img_path = '/home/vkorneychuk/projects/lum/photo/27.10/white/{}/{}'
max = 15
names = ['1-1-30-snapshot.png', '1-1-120-snapshot.png', '1-1-200-snapshot.png', '1-1-330-snapshot.png',
         '1-2-0-snapshot.png', '1-2-150-snapshot.png', '1-2-210-snapshot.png', '1-2-310-snapshot.png',
         '2-1-30-snapshot.png', '2-1-170-snapshot.png', '2-1-200-snapshot.png', '2-1-340-snapshot.png',
         '2-2-0-snapshot.png', '2-2-60-snapshot.png', '2-2-140-snapshot.png', '2-2-255-snapshot.png']
curr = max
img = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_COLOR)
img_gray = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_GRAYSCALE)

img_blank = np.zeros((img.shape[1], img.shape[0]), np.uint8)
cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.imshow('test', img)

while True:

    # считываем значения слайдеров
    kernel_side = cv.getTrackbarPos('Blur kernel', sett_def_win)
    dk = cv.getTrackbarPos('Dilate kernel', sett_def_win)
    g = cv.getTrackbarPos('Green', sett_def_win)
    b = cv.getTrackbarPos('Blue', sett_def_win)
    gamma = cv.getTrackbarPos('Gamma', sett_def_win) / 100 - 1
    thresh_min_hsv = cv.getTrackbarPos('Threshold min hsv', sett_def_win)
    thresh_max_hsv = cv.getTrackbarPos('Threshold max hsv', sett_def_win)
    # считываем значения слайдеров hsv
    h1 = cv.getTrackbarPos('h1', sett_def_win)
    s1 = cv.getTrackbarPos('s1', sett_def_win)
    v1 = cv.getTrackbarPos('v1', sett_def_win)
    h2 = cv.getTrackbarPos('h2', sett_def_win)
    s2 = cv.getTrackbarPos('s2', sett_def_win)
    v2 = cv.getTrackbarPos('v2', sett_def_win)

    # Apply gamma correction
    img_gamma = apply_gamma(img, gamma)
    # Apply blur
    blur_img = dbu.blur_img(img_gamma, kernel_side)
    # Apply hsv threshgold
    # img_hsv_res = dbu.hsv_threshold(blur_img, blur_img, hsv_min, hsv_max)
    # img_hsv_res_lights = dbu.hsv_threshold(blur_img, blur_img, hsv_min, hsv_max)
    # img_hsv_res |= img_hsv_res_lights
    # thresh_img_hsv, denoise_blade_hsv = dbu.denoise(img_hsv_res, img, thresh_min_hsv)
    # thresh_img_hsv, denoise_blade_hsv = dbu.denoise(img_hsv_res_lights, img, thresh_min_hsv)
    img_gray = cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY)

    # blue_channel = img_gamma[:, :, 0]
    # green_channel = img_gamma[:, :, 1]
    # red_channel = img_gamma[:, :, 2]
    #
    # b_diff = np.abs(img_gray - blue_channel)
    # g_diff = np.abs(img_gray - green_channel)
    # r_diff = np.abs(img_gray - red_channel)

    # bc = np.mean(img[:, 0])
    # gc = np.mean(img[:, 1])
    # rc = np.mean(img[:, 2])
    # cc = np.array([bc, gc, rc])
    # # cc = cc / np.sqrt((cc**2).sum())  # normalise the light (you might want to play with this a bit
    # img_norm = (img / cc).astype('uint8')
    # # res = fourie(img_gray)
    # cv.imshow('test', img_norm)

    # kernel = np.ones((dk, dk), 'uint8')
    # erode_blue = cv.erode(b_diff, kernel, iterations=1)
    # erode_green = cv.erode(g_diff, kernel, iterations=1)
    # erode_red = cv.erode(r_diff, kernel, iterations=1)

    # erode_blue = (255 - erode_blue)
    # erode_green = (255 - erode_green)
    # erode_red = (255 - erode_red)

    # cv.imshow(defect_win, dbu.stackImages([[img, erode_blue],
    #                                         [erode_green, erode_red]],1))
    # cv.imshow(defect_win, dbu.stackImages([[img, b_diff],
    #                                        [g_diff, r_diff]], 1))
    pushed_key = cv.waitKey(1)
    # Enter
    if pushed_key == 13:
        print('start')
        Z = img.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        cv.imshow(defect_win, res2)
        print('stop')
    elif pushed_key == 27:
        break
    elif pushed_key == 83:
        curr = curr + 1 if curr < max else 0
        img = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_COLOR)
        img_gray = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_GRAYSCALE)
        cv.imshow('test', img)
        print(names[curr])
    elif pushed_key == 81:
        curr = curr - 1 if curr > 0 else max
        img = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_COLOR)
        img_gray = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_GRAYSCALE)
        cv.imshow('test', img)
        print(names[curr])
    elif pushed_key != -1:
        print(pushed_key)