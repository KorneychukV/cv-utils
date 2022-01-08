import sys
import cv2
import neoapi
import numpy as np
import time

current_exposure = 130000

try:

    camera_one = neoapi.Cam()
    camera_two = neoapi.Cam()

    # serial number camera_one IP = 192.168.10.55
    sn_one = '700005666415'
    # serial number camera_two IP = 192.168.0.56
    sn_two = '700005666415'

    camera_one.Connect(sn_one)
    # camera_two.Connect(sn_two)

    camera_one.f.ExposureTime.Set(current_exposure)
    # camera_two.f.ExposureTime.Set(current_exposure)

    main_window = 'Press [ESC] to exit ..'
    cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)

    camera_one_index = 0
    camera_two_index = 0

    camera_one.f.TriggerMode.value = neoapi.TriggerMode_Off  # set camera to trigger mode
    # camera_two.f.TriggerMode.value = neoapi.TriggerMode_On  # set camera to trigger mode

    while True:
        img_one = camera_one.GetImage().GetNPArray()
        # print(img_one.shape)
        img_h, img_w, _ = img_one.shape
        img_one = cv2.circle(img_one, (int(img_w/2), int(img_h/2)), radius=15, color=(0, 0, 255), thickness=15)
        img = np.reshape(img_one, img_one.shape[:2])
        cv2.imshow(main_window, img_one)
        # cv2.imshow(main_window, img_two)

        pushed_key = cv2.waitKey(1)
        if pushed_key == 13:
            camera_one.f.TriggerSoftware.Execute()  # execute a software trigger to get an image
            img_one = camera_one.GetImage().GetNPArray()  # retrieve the image to work with it
            print(img_one.shape)
            cv2.imwrite('camera1_' + str(camera_one_index) + '.bmp', img_one)
            camera_one_index += 1
            # camera_two.f.TriggerSoftware.Execute()  # execute a software trigger to get an image
            # img_two = camera_two.GetImage().GetNPArray()  # retrieve the image to work with it
            # cv2.imwrite('camera2_' + str(camera_two_index) + '.bmp', img_two)
            # camera_two_index += 1
            print('images saved')
        elif pushed_key == 27:
            camera_one.f.TriggerMode.value = neoapi.TriggerMode_Off  # set camera to trigger mode
            # camera_two.f.TriggerMode.value = neoapi.TriggerMode_Off  # set camera to trigger mode
            break

except (neoapi.NeoException, Exception) as exc:
    print('error: ', exc.GetDescription())
    result = 1
