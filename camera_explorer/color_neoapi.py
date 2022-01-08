import sys
import cv2
import neoapi
import cv2 as cv
import numpy as np
import time

if __name__ == '__main__':
    def nothing(*arg):
        pass


current_exposure = 130000
iso = 1
thresh_val = 20
gamma = 99
main_win = 'main_window'
hsv_win = 'blade_range_window'
mask_win = 'blade_mask_range_window'
settings_win = 'settings'
cam_settings_win = 'cam_settings'
palitra_win = 'palitra'
gray_win = 'gray'

cv.namedWindow(main_win, cv.WINDOW_NORMAL)
cv.namedWindow(palitra_win, cv.WINDOW_NORMAL)
cv.namedWindow(hsv_win, cv.WINDOW_NORMAL)
cv.namedWindow(mask_win, cv.WINDOW_NORMAL)
cv.namedWindow(settings_win, cv.WINDOW_NORMAL)
cv.namedWindow(cam_settings_win, cv.WINDOW_NORMAL)
cv.namedWindow(gray_win, cv.WINDOW_NORMAL)

try:
    camera_one = neoapi.Cam()
    # serial number camera_one IP = 192.168.10.55
    sn_one = '700005000995'
    camera_one.Connect(sn_one)
    camera_one.f.ExposureTime.Set(current_exposure)
    camera_one.f.Gain.Set(iso)
    camera_one_index = 0
    camera_one.f.TriggerMode.value = neoapi.TriggerMode_Off  # set camera to trigger modee
    camera_one.f.BalanceWhiteAuto.Set(1)
    # camera_one.f.Gamma.Set(gamma/100)
    camera_one.f.ColorTransformationAuto.Set(0)

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    # color settings
    cv.createTrackbar('h1', settings_win, 37, 255, nothing)
    cv.createTrackbar('s1', settings_win, 8, 255, nothing)
    cv.createTrackbar('v1', settings_win, 0, 255, nothing)
    cv.createTrackbar('h2', settings_win, 76, 255, nothing)
    cv.createTrackbar('s2', settings_win, 255, 255, nothing)
    cv.createTrackbar('v2', settings_win, 255, 255, nothing)
    # cam settings
    cv.createTrackbar('gamma', cam_settings_win, gamma, 100, nothing)
    cv.createTrackbar('wb_element', cam_settings_win, 0, 8, nothing)

    img_palitra = cv.imread('./circle_rgb.jpg', cv.IMREAD_COLOR)

    current_wb_matrix_px = 0
    while True:
        img = camera_one.GetImage().Convert('BGR8').GetNPArray()
        cv.imshow(main_win, img)

        # считываем значения бегунков
        h1 = cv.getTrackbarPos('h1', settings_win)
        s1 = cv.getTrackbarPos('s1', settings_win)
        v1 = cv.getTrackbarPos('v1', settings_win)
        h2 = cv.getTrackbarPos('h2', settings_win)
        s2 = cv.getTrackbarPos('s2', settings_win)
        v2 = cv.getTrackbarPos('v2', settings_win)
        gamma = cv.getTrackbarPos('gamma', cam_settings_win)
        current_wb_matrix_px = cv.getTrackbarPos('wb_element', cam_settings_win)

        # формируем начальный и конечный цвет фильтра
        hsv_min = np.array((h1, s1, v1), np.uint8)
        hsv_max = np.array((h2, s2, v2), np.uint8)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # удаляем световой шум
        _, thresh_img = cv.threshold(img_gray, thresh_val, 255, 0)
        denoise_blade = cv.bitwise_and(img, img, mask=thresh_img)


        img_hsv = cv.cvtColor(denoise_blade, cv.COLOR_BGR2HSV)
        palitra_hsv = cv.cvtColor(img_palitra, cv.COLOR_BGR2HSV)

        # накладываем фильтр на кадр в модели HSV
        hsv_thresh = cv.inRange(img_hsv, hsv_min, hsv_max)
        hsv_palitra = cv.inRange(palitra_hsv, hsv_min, hsv_max)
        # Оставляем зелёные свечения по маске
        res = cv.bitwise_and(img_palitra, img_palitra, mask=hsv_palitra)
        res_blade = cv.bitwise_and(img, img, mask=hsv_thresh)

        # Выводим изображения
        cv.imshow(hsv_win, hsv_thresh)
        cv.imshow(mask_win, res_blade)
        cv.imshow(palitra_win, res)
        cv.imshow(gray_win, denoise_blade)

        pushed_key = cv.waitKey(1)
        # Enter
        if pushed_key == 13:
            cv2.imwrite('camera1_' + str(camera_one_index) + '.bmp', img)
            camera_one_index += 1
            print('images saved')
        # w
        elif pushed_key == 119:
            # wb matrix
            #ct_arr = [1.14014, -0.117188, -0.0229492, -0.0600586, 1.08301, -0.0229492, -0.0600586]
            print(camera_one.f.ColorTransformationEnable.Get())
            camera_one.f.ColorTransformationAuto.Set(1)
            camera_one.f.ColorTransformationFactoryListSelector.Set(1)
            camera_one.f.ColorTransformationValueSelector.Set(current_wb_matrix_px)
            val = camera_one.f.ColorTransformationValue.Get() - 0.5
            if val < -1:
                val = -1
            camera_one.f.ColorTransformationValue.Set(val)
            print(camera_one.f.ColorTransformationValue.Get())
        # e
        elif pushed_key == 101:
            # wb matrix
            print(camera_one.f.ColorTransformationEnable.Get())
            camera_one.f.ColorTransformationAuto.Set(1)
            camera_one.f.ColorTransformationFactoryListSelector.Set(1)
            camera_one.f.ColorTransformationValueSelector.Set(current_wb_matrix_px)
            val = camera_one.f.ColorTransformationValue.Get() + 0.5
            if val > 3:
                val = 3
            camera_one.f.ColorTransformationValue.Set(val)
            print(val)
        # g
        elif pushed_key == 103:
            camera_one.f.Gamma.Set(float(gamma)/100)
            print("Gamma = {}".format(gamma))
        # +
        elif pushed_key == 43:
            iso += 0.5
            camera_one.f.Gain.Set(iso)
            print('ISO = {}'.format(iso))
        # -
        elif pushed_key == 45:
            iso -= 0.5
            camera_one.f.Gain.Set(iso)
            print('ISO = {}'.format(iso))
        # *
        elif pushed_key == 42:
            current_exposure += 1000
            camera_one.f.ExposureTime.Set(current_exposure)
            print('Exposure = {}'.format(current_exposure))
        # /
        elif pushed_key == 47:  # /
            current_exposure -= 1000
            camera_one.f.ExposureTime.Set(current_exposure)
            print('Exposure = {}'.format(current_exposure))
        # i
        elif pushed_key == 105:
            thresh_val += 5
            print('Thresh min = {}'.format(thresh_val))
        # d
        elif pushed_key == 100:
            thresh_val -= 5
            print('Thresh min = {}'.format(thresh_val))
        # Esc
        elif pushed_key == 27:
            camera_one.f.TriggerMode.value = neoapi.TriggerMode_Off  # set camera to trigger mode
            break
        # else
        elif pushed_key != -1:
            print(pushed_key)

except (neoapi.NeoException, Exception) as exc:
    print('error: ', exc.GetDescription())
    result = 1
