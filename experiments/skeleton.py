import cv2 as cv
import numpy as np

cv.namedWindow('win', cv.WINDOW_NORMAL)
cv.namedWindow('src', cv.WINDOW_NORMAL)
cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.namedWindow('res_1', cv.WINDOW_NORMAL)
cv.namedWindow('scelet', cv.WINDOW_NORMAL)
cv.namedWindow('scelet_1', cv.WINDOW_NORMAL)
cv.namedWindow('blur', cv.WINDOW_NORMAL)
cv.namedWindow('cont', cv.WINDOW_NORMAL)


# image = cv.imread("spiral.jpg", cv.IMREAD_GRAYSCALE)
image = cv.imread("bublik.jpg", cv.IMREAD_GRAYSCALE)
# image = cv.imread("sharik.jpg", cv.IMREAD_GRAYSCALE)
# image = cv.imread("squares.jpg", cv.IMREAD_GRAYSCALE)
thresh = 90
thresh = 145
# thresh = 70
# thresh = 5

k = 20

blur = cv.blur(image, (k,k))
_, img = cv.threshold(blur, thresh, 255, cv.THRESH_BINARY_INV)
cnt, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
thinned = cv.ximgproc.thinning(img, thinningType = cv.ximgproc.THINNING_ZHANGSUEN)
thinned_1 = cv.ximgproc.thinning(img, thinningType = cv.ximgproc.THINNING_GUOHALL)
res = cv.addWeighted(img, 0.5, thinned, 1, 0)
res_1 = cv.addWeighted(img, 0.5, thinned_1, 1, 0)
contours, _ = cv.findContours(thinned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros_like(img)
cv.drawContours(blank, contours, -1, (255,255,255), 1)

while True:

    cv.imshow('win', image)
    cv.imshow('src', img)
    cv.imshow('scelet', thinned)
    cv.imshow('scelet_1', thinned_1)
    cv.imshow('res', res)
    cv.imshow('res_1', res_1)
    cv.imshow('blur', blur)
    cv.imshow('cont', blank)

    pushed_key = cv.waitKey(1)
    # Enter
    if pushed_key == 43:
        thresh += 5
        _, img = cv.threshold(blur, thresh, 255, cv.THRESH_BINARY_INV)
        print(thresh)
    elif pushed_key == 45:
        thresh -= 5
        _, img = cv.threshold(blur, thresh, 255, cv.THRESH_BINARY_INV)
        print(thresh)
    elif pushed_key == 112:
        k += 1
        blur = cv.blur(image, (k, k))
        print(k)
    elif pushed_key == 109:
        k -= 1
        blur = cv.blur(image, (k, k))
        print(k)
    elif pushed_key == 13:
        cnt, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        np.save('cnt.npy', cnt)
        thinned = cv.ximgproc.thinning(img, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
        thinned_1 = cv.ximgproc.thinning(img, thinningType=cv.ximgproc.THINNING_GUOHALL)
        res = cv.addWeighted(image, 0.5, thinned, 1, 0)
        res_1 = cv.addWeighted(img, 0.5, thinned_1, 1, 0)
        contours, _ = cv.findContours(thinned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        blank = np.zeros_like(img)
        cv.drawContours(blank, contours, -1, (255, 255, 255), 1)
    elif pushed_key == 27:
        break
    elif pushed_key != -1:
        print(pushed_key)