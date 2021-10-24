# StereoVision_0.1

Projekt wykorzystujący wizje geometryczną. Opierający się o rozpoznanie QR codu, wyznaczenie współrzędnych geometrycznych pudełka 3D, a następnie przeniesienie ich do wymiarów 2D dla obrazów z dwóch kamer znajdujących się pod kątem 90 stopni względem siebie.

Stanowisko z dwoma kamerami:

![alt text](https://github.com/OlszPrzem/ComputerVision/blob/master/StereoVision_0.1/img_add/widok_stanowiska.jpg?raw=true)

Widok z głównej kamery która odczytuje QR code, na podstawie którego sa wyznaczane wspołrzędne wierzchołków pudełka w 3D

![alt text](https://github.com/OlszPrzem/ComputerVision/blob/master/StereoVision_0.1/img_add/Image1_.png?raw=true)

Widok z drugiej kamery na obrazie której zmapowano wierzchołki pudełka z wczesniej wyznaczonych współrzędnych 3D. Dodatkowo, na przedniej ścianie pudełka, w miejscu kodu, jest wyświetlana grafika przy zachowaniu odpowiedniej perspektywy.

![alt text](https://github.com/OlszPrzem/ComputerVision/blob/master/StereoVision_0.1/img_add/Image2.png?raw=true)


Przykład zdjęcia z procesu kalibracji kamer

![alt text](https://github.com/OlszPrzem/ComputerVision/blob/master/StereoVision_0.1/img_add/calibracja.png?raw=true)


