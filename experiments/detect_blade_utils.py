import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
from math import sqrt


def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv.FILLED)
                cv.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

# Убираем шум на фоне
def denoise(for_mask_img, changing_img, thresh_min):
    img_gray = cv.cvtColor(for_mask_img, cv.COLOR_BGR2GRAY)
    _, thresh_img = cv.threshold(img_gray, thresh_min, 255, cv.THRESH_BINARY)
    denoise_blade = cv.bitwise_and(changing_img, changing_img, mask=thresh_img)
    return thresh_img, denoise_blade

# Apply Mono mixer for image
def mono_mixer(temp_img, b, g, r):
    mix_coefs = np.array([(b-500.0)/100, (g-500.0)/100, (r-500.0)/100])
    mix_img = np.dot(temp_img[...,:3], mix_coefs)
    mix_img = np.array(np.where(mix_img < 10, mix_img*0, 255), dtype='uint8')
    mix_img = cv.bitwise_and(temp_img, temp_img, mask=mix_img)
    return mix_img

def blur_img(img, kernel_side):
    # Apply blur
    kernel_side = 1 if kernel_side < 1 else kernel_side * 2 - 1
    blur_kernel = (kernel_side, kernel_side)
    blur_img = cv.blur(img, blur_kernel)
    return blur_img

def hsv_threshold(for_mask_img, changing_img, hsv_min, hsv_max):
    hsv_img = cv.cvtColor(for_mask_img, cv.COLOR_BGR2HSV)
    mask_img = cv.inRange(hsv_img, hsv_min, hsv_max)
    res_img = cv.bitwise_and(changing_img, changing_img, mask=mask_img)
    return res_img

def bright_contr(img, brightness=255, contrast=127):
    brightness = int(brightness + (-255))
    contrast = int(contrast + (-127))
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        al_pha = (max - shadow) / 255
        ga_mma = shadow
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv.addWeighted(img, al_pha, img, 0, ga_mma)
    else:
        cal = img
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv.addWeighted(cal, Alpha, cal, 0, Gamma)
    return cal

def CannyThreshold(img, val):
    low_threshold = val
    detected_edges = cv.Canny(img, low_threshold, low_threshold*3, 3)
    mask = detected_edges != 0
    return mask

def cluster_threshold(thresh_img, orig_lights_wb, median_thresh):
    # Минимальный размер свечения
    # MIN_SIZE = 7
    lights = np.array(np.nonzero(thresh_img))
    lights_trans = lights.T
    # кластеризуем свечения
    model = DBSCAN(eps=sqrt(2), min_samples=9)
    yhat = model.fit_predict(lights_trans)
    # все кластера
    clusters_with_non_cluster = np.unique(yhat)
    res = np.zeros_like(thresh_img)
    # Получаем медианное значение яркости каждого кластера (можно попробовать максимально значение яркости)

    print('Cluster size = {}'.format(len(clusters_with_non_cluster)))
    for index in clusters_with_non_cluster:
        if index != -1:
            cluster_indices = np.where(yhat == index)
            # получаем координаты точек кластера
            cluster_points = lights_trans[cluster_indices]
            cluster_points = cluster_points.T
            wmin = np.min(cluster_points[0, :])
            hmin = np.min(cluster_points[1, :])
            wmax = np.max(cluster_points[0, :])
            hmax = np.max(cluster_points[1, :])
            x = wmax - wmin
            y = hmax - hmin
            # если свечение маленькое, значит это шум
            # if sqrt(x**2 + y**2) > MIN_SIZE:
            cluster_bright = np.median(orig_lights_wb[cluster_points[0], cluster_points[1]])
            # оставляем только те кластера, медианная яркость которых выше пороговой
            if cluster_bright > median_thresh:
                print(cluster_bright)
                print('Cluster {}'.format(index))
                res[cluster_points[0], cluster_points[1]] = 255
    return res

def apply_gamma(img, gamma):
    gamma = gamma if gamma != 0 else 0.01
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(img, table)