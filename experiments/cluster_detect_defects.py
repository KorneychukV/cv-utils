import math

import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import detect_blade_utils as dbu

def nothing(*arg):
    pass

cv.namedWindow('test', cv.WINDOW_NORMAL)
cv.namedWindow('test1', cv.WINDOW_NORMAL)
cv.namedWindow('test2', cv.WINDOW_NORMAL)
cv.namedWindow('test3', cv.WINDOW_NORMAL)
cv.namedWindow('set', cv.WINDOW_NORMAL)
cv.namedWindow('palitra', cv.WINDOW_NORMAL)

inspection_id = 735
img_path = '/var/lib/docker/volumes/docker_media/_data/LEAP-1B STEP 2/{}/{}'
names = ['1-1-30-snapshot.png', '1-1-120-snapshot.png', '1-1-200-snapshot.png', '1-1-330-snapshot.png',
         '1-2-0-snapshot.png', '1-2-150-snapshot.png', '1-2-210-snapshot.png', '1-2-310-snapshot.png',
         '2-1-30-snapshot.png', '2-1-170-snapshot.png', '2-1-200-snapshot.png', '2-1-340-snapshot.png',
         '2-2-0-snapshot.png', '2-2-60-snapshot.png', '2-2-140-snapshot.png', '2-2-255-snapshot.png']
max = 15
curr = max

# img = cv.imread('../../lum/photo/presentation/{}.png'.format(name), cv.IMREAD_COLOR)
img = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_COLOR)
img = np.where(img == [255, 255, 255], [254, 255, 254], img).astype('uint8')

h1 = 16
s1 = 0
v1 = 0
h2 = 75
s2 = 255
v2 = 255
# median_thresh = 57
median_thresh = 45
gamma = 10
thresh_min = 45

img_palitra = cv.imread('./circle_rgb.jpg', cv.IMREAD_COLOR)
img_palitra = np.where(img_palitra == [255, 255, 255], [254, 255, 254], img_palitra).astype('uint8')

cv.createTrackbar('Gamma', 'set', gamma, 30, nothing)
cv.createTrackbar('h1', 'set', h1, 255, nothing)
cv.createTrackbar('s1', 'set', s1, 255, nothing)
cv.createTrackbar('v1', 'set', v1, 255, nothing)
cv.createTrackbar('h2', 'set', h2, 255, nothing)
cv.createTrackbar('s2', 'set', s2, 255, nothing)
cv.createTrackbar('v2', 'set', v2, 255, nothing)

while True:

    gamma = cv.getTrackbarPos('Gamma', 'set') / 10
    h1 = cv.getTrackbarPos('h1', 'set')
    s1 = cv.getTrackbarPos('s1', 'set')
    v1 = cv.getTrackbarPos('v1', 'set')
    h2 = cv.getTrackbarPos('h2', 'set')
    s2 = cv.getTrackbarPos('s2', 'set')
    v2 = cv.getTrackbarPos('v2', 'set')
    hsv_min = (h1, s1, v1)
    hsv_max = (h2, s2, v2)

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
    cv.imshow('test3', thresh_img)
    cv.imshow('test1', res)
    # 1-2-310

    pushed_key = cv.waitKey(1)
    # Enter
    if pushed_key == 43:
        thresh_min += 5
        print(thresh_min)
    # r
    elif pushed_key == 114:
        img = np.rot90(img, 1)
    elif pushed_key == 45:
        thresh_min -= 5 if thresh_min > 0 else 0
        print(thresh_min)
    elif pushed_key == 83:
        curr = curr+1 if curr < max else 1
        img = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_COLOR)
        img = np.where(img == [255, 255, 255], [254, 255, 254], img).astype('uint8')
        print(names[curr])
    elif pushed_key == 81:
        curr = curr-1 if curr > 0 else max
        img = cv.imread(img_path.format(inspection_id, names[curr]), cv.IMREAD_COLOR)
        img = np.where(img == [255, 255, 255], [254, 255, 254], img).astype('uint8')
        print(names[curr])
    elif pushed_key == 82:
        median_thresh += 2
        print(median_thresh)
    elif pushed_key == 84:
        median_thresh -= 2 if median_thresh > 0 else 0
        print(median_thresh)
    elif pushed_key == 13:
        res = dbu.cluster_threshold(thresh_img, orig_lights_wb, median_thresh)
        res = cv.addWeighted(img, 1.0, cv.cvtColor(res, cv.COLOR_GRAY2BGR), 0.5, 1)
        cv.imshow('test2', res)
    elif pushed_key == 27:
        break
    elif pushed_key != -1:
        print(pushed_key)
