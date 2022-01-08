# time to photo
import sys
import cv2
import neoapi
import os
import numpy as np
import time
import matplotlib.pyplot as plt

results = []

result_window = 'Result'
cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)

try:
    camera = neoapi.Cam()
    camera.Connect()
    camera.f.ExposureTime.Set(50000)
    photo_count = 0
    while photo_count < 50:
        if cv2.waitKey(1) == 13:
            start_time = time.time()
            img = camera.GetImage().GetNPArray()
            cv2.imwrite(os.path.join('ttp_result','photo.bmp'), img)
            end_time = time.time()
            results.append(end_time-start_time)

            photo_count += 1
            print(str(photo_count) + ' ' + str(end_time-start_time))

except (neoapi.NeoException, Exception) as exc:
    print('error: ', exc.GetDescription())

plt.plot(results)
plt.ylabel('ttp, sec')
plt.show()

a = np.array(results)
print(np.median(a))