import serial
import time
import neoapi

# sudo chown vkorneychuk /dev/ttyUSB0

stage = 0

ser = serial.Serial('/dev/ttyUSB0', 57600, timeout=5)
packet = bytearray()
packet.append(0x65)
packet.append(0x40)

camera_one = neoapi.Cam()
sn_one = '700005666414'
camera_one.Connect(sn_one)
current_exposure = 99000
camera_one.f.ExposureTime.Set(current_exposure)
camera_one_index = 0
camera_one.f.TriggerMode.value = neoapi.TriggerMode_On

start_time = 0

while (True):


    if stage == 0:
        if start_time < (time.time() - 0.5):
            start_time = time.time()
            camera_one.f.TriggerSoftware.Execute()  # execute a software trigger to get an image
            img_one = camera_one.GetImage().Convert('RGB8')  # retrieve the image to work with it
            img_one.Save('camera1_' + str(camera_one_index) + '.bmp')
            camera_one_index += 1
            print("{} written!".format('camera1_' + str(camera_one_index) + '.bmp'))
            print(time.time())
            stage = 1
    elif stage == 1:
        ser.flushInput()
        ser.write(packet)
        stage = 3
    elif stage == 2:
        if (b'K' == ser.read()):
            stage = 3
    elif stage == 3:
        if camera_one_index <= 7:
            stage = 0
        else:
            break

camera_one.f.TriggerMode.value = neoapi.TriggerMode_Off

