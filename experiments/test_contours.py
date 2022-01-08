import cv2 as cv
import  numpy as np
from time import time

cv.namedWindow('src', cv.WINDOW_NORMAL)
cv.namedWindow('thin', cv.WINDOW_NORMAL)
cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.namedWindow('cnt', cv.WINDOW_NORMAL)

# точки контура
points = np.int32([[100,100], [600,100], [600,500], [100,500], [300,200], [500,200], [500,500], [300,500]])
start = time()

# создаём картинку под размеры контура
# сдвигаем контур в ноль
x_min = np.min(points[:,0])
y_min = np.min(points[:,1])
points -= [x_min-10, y_min-10]
shape = (np.max(points[:,1]+10), np.max(points[:,0]+10), 1)

# Делаем сужение
img = np.full(shape, 0, dtype='uint8')
img = cv.fillPoly(img, [points], 255)
thinned = cv.ximgproc.thinning(img, thinningType = cv.ximgproc.THINNING_ZHANGSUEN)

# Высчитываем длину через половину периметра контура
cnt, _ = cv.findContours(thinned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
length = cv.arcLength(cnt[0], False)/2

end = time()

print('length {}\ntime {}'.format(length, end-start))
blank = np.zeros_like(img)
cv.drawContours(blank, cnt, -1, (255,255,255), 1)
res = cv.addWeighted(img, 0.5, thinned, 1, 0)

cv.imshow('src', img)
cv.imshow('thin', thinned)
cv.imshow('res', res)
cv.imshow('cnt', blank)

cv.waitKey()