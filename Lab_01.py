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
video_path = 'video_2.mp4'
#video_path = 'Lab_01\\video_pendulo.mp4'  # Cambia esto al camino de tu video
cap = cv2.VideoCapture(video_path)

# Crear una ventana y configurar la función de callback de mouse
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)

nuevo_alto = 480
nuevo_ancho = 680
#este primero es para el video pequeño
lower = np.array([0,69,0])
upper = np.array([138,255,255])
#lower = np.array([107,0,0])
#upper = np.array([128,255,255])
#lower = np.array([107,0,0])
#upper = np.array([128,255,239])

delta_t = 1/cap.get(cv2.CAP_PROP_FPS)
centros_x = []
centros_y = []

i = -1 # ti = delta_t * i esta formula nos dice, el ti (tiempo en el que se capturo el frame i)
# empezamos en menos 1 ya que al entrar al ciclo while aumentaremos a t=0s
while True:
    
    
    if not pausar_video:
        ret, frame = cap.read()
        #frame2 = np.zeros(frame.shape, dtype=np.uint8)
        frame2 = np.copy(frame)
        i+=1
        if not ret:
            break
        
    frame_alt = cv2.resize(np.copy(frame2), (nuevo_ancho, nuevo_alto))
    hsv = cv2.cvtColor(frame_alt, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
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
    
    print(f"Coordenadas del centro de masa ({xc:.0f}, {yc:.0f})")
    print()
    
    centros_x.append(xc)
    centros_y.append(yc)
    
    if len(centros_x) >= 2 and len(centros_y) >= 2: # necesitamos minimos 2 puntos para poder hallar
                                                    # la velocidad (Derivada de la posicion)
        if len(centros_x) >= 3 and len(centros_y)>= 3:
            
            vx = np.gradient(centros_x, delta_t) # velocidad de x es la derivada de la posicion respecto al tiempo
            vy = np.gradient(centros_y, delta_t) # velocidad de y es la derivada de la posicion respecto al tiempo
            ax = np.gradient(vx, delta_t) # acelereacion en x
            ay = np.gradient(vy, delta_t) # aceleracion en y
            
            # Actualmente estamos en unidades de px/s y px/s^2
            # Por lo que ahora haremos la conversion de px/s y px/s^2 a m/s y m/s^2
            # hallando la relacion, 1 px = 3.629e-4
            
            vx = vx*3.629e-4 
            vy = vy*3.629e-4
            ax = ax*3.629e-4
            ay = ay*3.629e-4
                        
            mv = np.sqrt(vx**2 + vy**2) # magnitud de la velocidad
            ma = np.sqrt(ax**2 + ay**2) # magnitud de la aceleracion
            
            print(f"La velocidad de x en el tiempo {i*delta_t:.2f} segundos es de {vx[-1]:.2f} m/s")
            # debe ser vx[-1] ya que debemos hallar la velocidad en ese ultimo punto que acabamos
            # de agregar al vector de centros
            print()
            print(f"La velocidad de y en el tiempo {i*delta_t:.2f} segundos es de {vy[-1]:.2f} m/s")
            print()
            print(f"La aceleracion de x en el tiempo {i*delta_t:.2f} segundos es de {ax[-1]:.2f} m/s^2")
            print()
            print(f"La aceleracion de y en el tiempo {i*delta_t:.2f} segundos es de {ay[-1]:.2f} m/s^2")
            print()
            print(f"La magnitud de la velocidad en el tiempo {i*delta_t:.2f} segundos es de {mv[-1]:.2f}")
            print()
            print(f"La magnitud de la aceleracion en el tiempo {i*delta_t:.2f} segundos es de {ma[-1]:.2f}")
            print()
    
            # para dibujar el vector de la velocidad
            theta_v = np.arctan2(vy, vx) # calcula theta = la tangente inversa de la velocidad en x sobre la velocidad en y
            dvx = np.cos(theta_v) # encontramos el desplazamiento del vector de velocidad en x
            dvy = np.sin(theta_v) # encontramos el desplazamiento del vector de velocidad en y
            
            # para dibujar el vector de la aceleracion
            theta_a = np.arctan2(ay, ax) # calcula theta = la tangente inversa de la velocidad en x sobre la velocidad en y
            dax = np.cos(theta_a) # encontramos el desplazamiento del vector de aceleracion en x
            day = np.sin(theta_a) # encontramos el desplazamiento del vector de aceleracion en y
            
            
    
    # mostrar texto del color del pixel sin alterar la imagen usada para el calculo del centro de masa
    if mostrar_pixel:
        pix_col = mask_erode[y, x]  
        mask_vis = cv2.cvtColor(mask_erode, cv2.COLOR_GRAY2BGR) # para que se pueda mostrar el texto (3 canales)
        cv2.putText(mask_vis, f'Posicion: ({x}, {y}) Intensidad: {pix_col}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.circle(mask_vis, (int(xc),int(yc)), 3, (0,0,255), -1) # dibujar el centro de masa
        
        cv2.putText(mask_vis, f"Centro de masa: ({xc:.0f}, {yc:.0f})", (10,350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        # dibujar el vector
        cv2.arrowedLine(mask_vis,(int(xc),int(yc)), (int(xc+150*mv[-1]*dvx[-1]),int(yc+150*mv[-1]*dvy[-1])), (255,0,0), thickness=2)
        cv2.putText(mask_vis, f"Vx = {vx[-1]:.2f}, Vy = {vy[-1]:.2f} Magnitud velocidad: {mv[-1]:.2f}", (10,450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.arrowedLine(mask_vis,(int(xc),int(yc)), (int(xc+50*ma[-1]*dax[-1]),int(yc+50*ma[-1]*day[-1])), (0, 0,151), thickness=2,line_type=8)
        cv2.putText(mask_vis, f"ax = {ax[-1]:.2f}, ay = {ay[-1]:.2f} Magnitud aceleracion: {ma[-1]:.2f}", (10,400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow('Video', mask_vis)
    else:
        mask_vis = cv2.cvtColor(mask_erode, cv2.COLOR_GRAY2BGR) # para que se pueda mostrar el texto (3 canales)
        cv2.circle(mask_vis, (int(xc),int(yc)), 3, (0,0,255), -1)
        cv2.putText(mask_vis, f"Centro de masa: ({xc:.0f}, {yc:.0f})", (10,350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        
        # para poder mostrar el vector de la velocidad y aceleracion, debemos asegurarnos de que existan almenos
        # 3 posiciones distintas captadas del centro de masa, para poder hallar la primera y segunda derivada
        # que serian la velocidad y aceleracion respectivamente y asi poder mostrar sus magnitudes
        
        if len(centros_x) >= 2 and len(centros_y) >= 3:
            cv2.arrowedLine(mask_vis,(int(xc),int(yc)), (int(xc+150*mv[-1]*dvx[-1]),int(yc+150*mv[-1]*dvy[-1])), (255,0,0), thickness=2)
            cv2.putText(mask_vis, f"Vx = {vx[-1]:.2f}, Vy = {vy[-1]:.2f} Magnitud velocidad: {mv[-1]:.2f}", (10,450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.arrowedLine(mask_vis,(int(xc),int(yc)), (int(xc+50*ma[-1]*dax[-1]),int(yc+50*ma[-1]*day[-1])), (0, 0,151), thickness=2,line_type=8)
            cv2.putText(mask_vis, f"ax = {ax[-1]:.2f}, ay = {ay[-1]:.2f} Magnitud aceleracion: {ma[-1]:.2f}", (10,400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow('Video', mask_vis)
    
    
    
    
    key = cv2.waitKey(33) & 0xFF
    if key == 27:  # Presiona la tecla 'Esc' para salir
        break
    
    
cap.release()
cv2.destroyAllWindows()