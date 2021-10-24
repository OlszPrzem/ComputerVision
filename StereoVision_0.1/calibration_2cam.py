import cv2
import numpy as np
import os
from pyzbar import pyzbar

def calibrate_chessboard(dir_path, image_format, square_size, width, height):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((1, height*width, 3), np.float32)
    objp[0,:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = os.listdir(dir_path)
    for fname in images:
        img = cv2.imread(str(f'{dir_path}{fname}'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, (9,7), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print('no')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def calibrate_chessboard_stereo(dir_path1, dir_path2, image_format, square_size, width, height):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((1, height * width, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size
    print('objp')
    print(objp)

    objpoints = []  # 3d point in real world space
    imgpoints1 = []  # 2d points in image plane.
    imgpoints2 = []

    images1 = os.listdir(dir_path1)
    images2 = os.listdir(dir_path2)

    images1.sort()
    images2.sort()

    for fname in images1:
        img = cv2.imread(str(f'{dir_path1}{fname}'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints1.append(corners2)

            cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(300)
        else:
            print(fname)
            print('no')

    for fname in images2:
        img = cv2.imread(str(f'{dir_path2}{fname}'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints2.append(corners2)

            cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(300)
        else:
            print(fname)
            print('no')


    # Calibrate camera
    print(f'gray.shape: {gray.shape[::-1]}')
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, (1280,720), None, None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, (1280,720), None, None)

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)

    return [retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F]

def project( points, rvec, tvec, camera_matrix, dist_coefs):
    '''
    Funkcja do przeksztalcenia wymiarów w 3D do 2D na obrazie
    '''
    rvec1 = np.array(rvec, np.float)
    tvec1 = np.array(tvec, np.float)
    camera_matrix1 = np.array(camera_matrix, np.float)
    dist_coefs1 = np.array(dist_coefs, np.float)
    image_points, jac = cv2.projectPoints(points.T.reshape(-1,1,3), rvec1, tvec1, camera_matrix1, dist_coefs1)
    arr = np.array([image_points.reshape(-1,2).T])

    if arr[0][0]> 50000 or arr[0][1]> 50000:
        arr[0][0]=50000.0
        arr[0][1]=50000.0
    if arr[0][0]< -50000 or arr[0][1]< -50000:
        arr[0][0]=-50000.0
        arr[0][1]=-50000.0
    return arr

#inicjalizacja obu kamer, oraz pobranie pierwszej klatki
cap1 = cv2.VideoCapture(0)
cap1.set(3,1280)
cap1.set(4,720)

ret1, img1 = cap1.read()

cap2 = cv2.VideoCapture(2)
cap2.set(3,1280)
cap2.set(4,720)

ret2, img2 = cap2.read()

# sciezki do folderow ze zdjeciami wykonanymi do kalibracji
IMAGES_DIR1 = 'D:\\Python\\Projekty\\stereovision\\calibration_camera1_stereo2_3\\'
IMAGES_DIR2  = 'D:\\Python\\Projekty\\stereovision\\calibration_camera2_stereo2_3\\'
IMAGES_FORMAT = '.png'
SQUARE_SIZE = 1.1 # wymiary czernych kwadratow na tablicy kalibracyjnej w cm
WIDTH = 8 # liczba kwadratow w poziomie, pomniejszona o 1 (odpowiada liczbie przeciec)
HEIGHT = 6 # analogicznie, tylko w pionie

# wymiary qr_codu (w cm)
obj_points_my = np.array([[0., 0., 0.],
                            [4.4,0., 0.],
                            [4.4,4.4, 0.],
                            [0,4.4, 0.]
                          ], dtype = np.float) #punkty na qr kodzie

#wymiary skrajnych punktow prostopadloscianu na ktorym jest naklejony qr_code
obj_points_my1 = np.array([[-1.7, -1.7, 0.], # przednia sciana
                            [-1.7, 6.8, 0.],
                            [6.7, 6.8, 0.],
                            [6.7, -1.7, 0.],

                            [-1.7, -1.7, -3.1], # tylna sciana
                            [-1.7, 6.8, -3.1],
                            [6.7, 6.8, -3.1],
                            [6.7, -1.7, -3.1],
                          ], dtype = np.float)

# tablica przechowujaca skrajne punkty prostopadloscianu, zmapowane dla kamery drugiej
obj_points_my2 = np.zeros((8,3), dtype =np.float)

# wczytanie obrazu ktory ma zostac wyswietlony na powierzchni prostopadloscianu, na obrazie z kamery 2
img_show = cv2.imread("bmw.jpg")
pts_src = np.array([[0,0],[img_show.shape[0], 0], [img_show.shape[0], img_show.shape[1]], [0, img_show.shape[1]]])

# kolory markerow w rogach prostopadloscianu
colors = [(255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0), (128, 0, 128), (0, 0, 128), (128, 0, 0), (0, 128, 0)]

if __name__ == "__main__":

    # wyliczenie macierzy rotacji i translacji kamery 1 wzgledem 2
    [retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_, T_, E, F] = calibrate_chessboard_stereo(
        IMAGES_DIR1,
        IMAGES_DIR2,
        IMAGES_FORMAT,
        SQUARE_SIZE,
        WIDTH,
        HEIGHT
        )

    print('Camera matrix:')
    print(cameraMatrix1)
    print('Dist. Coef. :')
    print(distCoeffs1)

    # rotacja punktow z kamery 1 na kamere 2
    for i in range(8):
        obj_points_my2[i] = R_ @ (obj_points_my1[i].reshape(-1, 1).ravel())

    while True:
        ret1, frame1 = cap1.read() # pobranie obrazu z kamer
        ret2, frame2 = cap2.read()

        barcodes = pyzbar.decode(frame1) # poszukiwanie gr_codu oraz jego dekodowaie

        if ret1 and ret2:
            if barcodes is not None:
                for barcode in barcodes:
                    if (len(barcode.polygon)>=3): # znalezienie gr_codu oraz sprawdzenie czy poligon ma 4 narozniki

                        # zaznaczenie niebieskim kwadratem miejsce wystepowania gr_codu
                        (x, y, w, h) = barcode.rect
                        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # wyswietlenie napisu co zostało zakodowane w qr_codzie
                        barcodeData = barcode.data.decode('utf-8')
                        barcodeType = barcode.type
                        text = "{} ( {} )".format(barcodeData, barcodeType)
                        cv2.putText(frame1, text, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

                        # konwersja kolejnosci punktow poligonu w zaleznosci od orientacji qr_codu
                        if barcode.polygon[3].x >= barcode.polygon[1].x:
                            corners2 = np.array(
                                [
                                    [barcode.polygon[0].x, barcode.polygon[0].y],
                                    [barcode.polygon[1].x, barcode.polygon[1].y],
                                    [barcode.polygon[2].x, barcode.polygon[2].y],
                                    [barcode.polygon[3].x, barcode.polygon[3].y]
                                ],
                                dtype=np.float)
                        else:
                            corners2 = np.array(
                                [
                                    [barcode.polygon[3].x, barcode.polygon[3].y],
                                    [barcode.polygon[0].x, barcode.polygon[0].y],
                                    [barcode.polygon[1].x, barcode.polygon[1].y],
                                    [barcode.polygon[2].x, barcode.polygon[2].y]
                                ],
                                dtype=np.float)

                        # wyswietlenie skrajnych punktow poligonu
                        for cor, _ in enumerate(corners2):
                            draw_point = (int(corners2[cor][0]), int(corners2[cor][1]))
                            cv2.drawMarker(frame1, draw_point, color=colors[cor], markerType=cv2.MARKER_CROSS,thickness=2)

                        # wyliczenie macierzy rotacji i translacji na podstawie znajomosci rzeczywistych wymiarow qr_codu oraz jego wymiarow na obrazie
                        _, rvecs_1, tvecs_1, _ = cv2.solvePnPRansac(obj_points_my, corners2, cameraMatrix1, distCoeffs1)#, flags=cv2.SOLVEPNP_P3P)

                        draw_point = None

                        # wyliczenie macierzy rotacji i translacji dla drugiej kamery, z wykorzystaniem wczesniej obliczonych macierzy dla stereokalibracji
                        rvecs_2 = np.matmul(R_, rvecs_1)
                        tvecs_2 = np.matmul(R_, tvecs_1) + T_

                        points_img_show = np.zeros((4,2), dtype=int)

                        for i in range(8):
                            # projekcja punktow dla kamery 1 i 2 ze wspolrzednych 3D do 2D
                            point2_1 = project(obj_points_my1[i], rvecs_1, tvecs_1, cameraMatrix1, distCoeffs1)
                            point2_2 = project(obj_points_my2[i], rvecs_2, tvecs_2, cameraMatrix2, distCoeffs2)

                            # wyrysowanie skrajnych punktow prostopadloscianu na klatkach obrazu
                            draw_point1 = (int(point2_1[0][0]), int(point2_1[0][1]))
                            draw_point2 = (int(point2_2[0][0]), int(point2_2[0][1]))

                            cv2.drawMarker(frame1, draw_point1, color=colors[i], markerType=cv2.MARKER_CROSS, thickness=2)
                            cv2.drawMarker(frame2, draw_point2, color=colors[i], markerType=cv2.MARKER_CROSS, thickness=2)

                            # przekształcenie 4 punktow z przedniej sciany prostopadloscianu, potrzebnych do przeksztalcenia dla obrazu ktory ma zostac wyswietlony
                            if i < 4:
                                points_img_show[i]=[int(point2_2[0][0]), int(point2_2[0][1])]

                        # wyznaczenie macierzy homograficznej pomiedzy dwoma obrazami
                        h, status = cv2.findHomography(pts_src, points_img_show, cv2.RANSAC)

                        # przeksztalcenie obrazu ktory ma zostać wyswietlony do wymiarow przedniej sciany
                        frame2_0 = cv2.warpPerspective(img_show, h, (frame2.shape[1], frame2.shape[0]))

                        # wyznaczenie maski do wyciecia zbednago tla z przeksztalconego obrazu
                        frame_gray = cv2.cvtColor(frame2_0, cv2.COLOR_BGR2GRAY) # zmiana kolorow
                        ret, mask = cv2.threshold(frame_gray, 127, 255, cv2.THRESH_BINARY) # binaryzacja
                        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # znajdowanie kontorow
                        mask = cv2.drawContours(mask, [max(contours, key = cv2.contourArea)], -1, 255, thickness=-1) # okreslenie najwiekszego konturu i wynaczenie koncowej maski

                        # polaczenie zdjec na podstawie wartosci maski
                        frame2[np.where(mask == 255)] = frame2_0[np.where(mask == 255)]

        # wyswietlenie uzyskanych obrazow po wszystkich operacjach
        if ret1:
            cv2.imshow("Image1", frame1)

        if ret2:
            cv2.imshow("Image2", frame2)

        cv2.waitKey(1)