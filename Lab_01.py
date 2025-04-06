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
upper = np.array([188,91,43])
lower = np.array([185,88,40])

def nothing(x):
    pass



# Crear una ventana con controles deslizantes (trackbars)
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)


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
    
    # Obtener valores de los trackbars
    # Leer valores de las trackbars
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")
    
    bajo = np.array([h_min, s_min, v_min])
    alto = np.array([h_max,s_max,v_max])
    
    mask = cv2.inRange(hsv, bajo,alto)
    
    cv2.imshow('video mascara', mask)
    cv2.imshow('Video', hsv)
    
    key = cv2.waitKey(33) & 0xFF
    if key == 27:  # Presiona la tecla 'Esc' para salir
        break
    
    
cap.release()
cv2.destroyAllWindows()
