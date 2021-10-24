import cv2

#incicjalizacja kamer
cap1 = cv2.VideoCapture(0)
cap1.set(3,1280)
cap1.set(4,720)

cap2 = cv2.VideoCapture(2)
cap2.set(3,1280)
cap2.set(4,720)


i = 0
n=0

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    cv2.imshow('frame1',frame1)
    cv2.imshow('frame2',frame2)
    cv2.waitKey(1)

    if n>40:
        cv2.imwrite(f'calibration_camera1_stereo2_3/picture_{i}.png',frame1)
        cv2.imwrite(f'calibration_camera2_stereo2_3/picture_{i}.png',frame2)
        n=0
        i+=1
        print(f'{i} photo')
    n+=1

    if i>40:
        break
