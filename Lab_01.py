import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Variables globales
pausar_video = False
mostrar_pixel = False

# Función de callback de mouse
def mouse_callback(event, _x, _y, flags, param):
    global pausar_video, mostrar_pixel, x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = _x, _y
        mostrar_pixel = True
    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video
        
# Cargar el video
video_path = 'Lab_01\\video_pendulo.mp4'  # Cambia esto al camino de tu video
cap = cv2.VideoCapture(video_path)

# Crear una ventana y configurar la función de callback de mouse
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)

nuevo_alto = 480
nuevo_ancho = 680
lower = np.array([107,0,0])
upper = np.array([128,255,239])




while True:
    if not pausar_video:
        ret, frame = cap.read()
        #frame2 = np.zeros(frame.shape, dtype=np.uint8)
        frame2 = np.copy(frame)
        if not ret:
            break
    if mostrar_pixel:
        pixel_color = frame[y, x]  # Obtener el valor del color en (x, y)
        frame2 = np.copy(frame)
        cv2.putText(frame2, f'Posicion: ({x}, {y}) Color: {pixel_color}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    frame3 = cv2.resize(frame2, (nuevo_ancho,nuevo_alto))
    hsv = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)

    
    mask = cv2.inRange(hsv, lower,upper)
    mask = 255-mask
    mask_erode = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    
    #calculando el centro de masa
    Y, X = np.indices(mask_erode.shape) # la primera matriz contiene los indices de fila y la segunda la de las columnas
    
    M = np.sum(mask_erode) # la masa seria la suma de todos los valores de intensidad en la region del objeto
                           # como el fondo es = 0, entonces podemos tomarlo, pues su valor no aporta nada
    sum_x = np.sum(X*mask_erode) #la sumatoria de la coordenada en x * la intensidad en ese pixel
    sum_y = np.sum(Y*mask_erode) #la sumatoria de la coordenada en y * la intensidad en ese pixel
    
    xc = 1/M * sum_x # Xc = la suma de las x dividido la masa
    yc = 1/M * sum_y # Yc = la suma de las y dividido la masa
    
    print(f"Coordenadas del centro de masa ({xc:.2f}, {yc:.2f})")
    print()
    #cv2.imshow('Video', mask)
    cv2.imshow('Video', mask_erode)
    
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # Presiona la tecla 'Esc' para salir
        break
    
    
cap.release()
cv2.destroyAllWindows()
