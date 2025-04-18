import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion() # grafico interactivo

#creamos grafico
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))

tiempos = [] # guardaremos el eje x (tiempos)
vel_tang = [] # velocidades tangenciales a graficar
ace_tang = [] # aceleraciones tangenciales  graficar

# obtenemos las lineas de las aceleraciones y velocidades para graficar
linea_v, = ax1.plot(tiempos, vel_tang, 'b-', label='Velocidad Tangencial (m/s)')
linea_a, = ax2.plot(tiempos, ace_tang, 'r-', label='Aceleración Tangencial (m/s²)')

# definimos los limites del grafico de la velocidad
ax1.set_xlim(0, 4)
ax1.set_ylim(-2, 2)
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Velocidad (m/s)')
ax1.legend() # mostramos el label de la linea_v

# definimos los limites del grafico de la aceleracion
ax2.set_xlim(0, 4)
ax2.set_ylim(-10, 10)
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Aceleración (m/s²)')
ax2.legend()

# Variables globales
pausar_video = False
mostrar_pixel = False

señal = 3 # para utilizarla en la convolucion como kernel o delta(n-k)

# Función de callback de mouse
def mouse_callback(event, _x, _y, flags, param):
    global pausar_video, mostrar_pixel, x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = _x, _y
        mostrar_pixel = True
    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video
        
# Cargar el video
video_path = 'Lab_01//video_pendulo.mp4'
cap = cv2.VideoCapture(video_path)

#crear una ventana y configurar la función de callback de mouse
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)
nuevo_alto = 480
nuevo_ancho = 680
lower = np.array([80,64,0])
upper = np.array([116,255,255])
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
            kernel = np.ones(señal)/señal # esta sera la señal para usar en la convolucion
            
            # usamos la convolucion para suavizar las velocidades y eliminar ruido
            vx_suavizado = np.convolve(vx, kernel, mode="same")
            vy_suavizado = np.convolve(vy, kernel, mode="same")
            
            # hallamos las aceleraciones
            ax = np.gradient(vx_suavizado, delta_t) # acelereacion en x
            ay = np.gradient(vy_suavizado, delta_t) # aceleracion en y
            
            # suavizamos las aceleraciones
            ax_suavizado = np.convolve(ax, kernel, mode="same")
            ay_suavizado = np.convolve(ay, kernel, mode="same")

            # Actualmente estamos en unidades de px/s y px/s^2
            # Por lo que ahora haremos la conversion de px/s y px/s^2 a m/s y m/s^2
            # hallando la relacion, 1 px = 6.7710e-4 metros
            vx_suavizado *= 6.7710e-4
            vy_suavizado *= 6.7710e-4
            ax_suavizado *= 6.7710e-4
            ay_suavizado *= 6.7710e-4
            mv = np.sqrt(vx_suavizado**2 + vy_suavizado**2) # magnitud de la velocidad
            ma = np.sqrt(ax_suavizado**2 + ay_suavizado**2) # magnitud de la aceleracion
            
            # ahora calculamos la aceleracion y velocidades tangenciales
            vt = mv*np.sign(vx_suavizado)#velocidad tangencial = magnitud de velocidad * signo de la velocidad en x
            at = ma*np.sign(ax_suavizado)
            
            # guardamos las velocidades y aceleraciones tangenciales para poder graficarlas
            vel_tang.append(vt[-1])
            ace_tang.append(at[-1])
            t_actual = i*delta_t # tiempo actual en segundos
            tiempos.append(t_actual)
            
            # actualizar los gráficos (x,y)
            linea_v.set_data(tiempos, vel_tang)
            linea_a.set_data(tiempos, ace_tang)
            
            # si el t_actual supera el esta cerca (1seg antes) del limite del grafico
            if t_actual > ax1.get_xlim()[-1] - 1:
                ax1.set_xlim(0,t_actual + 6) # actualizamos el eje x desde cero hasta t_actual + 6
                ax2.set_xlim(0,t_actual + 6)
                
            # ahora para el eje y
            # obtenemos los mayores valores de las velocidades y aceleraciones tangenciales
            # el 0.5 es para almenos tener un limite en donde graficar los ejes y
            max_v = max(np.max(np.abs(vel_tang)), 0.5)
            max_a = max(np.max(np.abs(ace_tang)), 0.5)
            
            # actualizamos consante mente los eje y dejando un margen de 0.2 y 0.5 para ver completamente el grafico
            ax1.set_ylim(-max_v - 0.2, max_v + 0.2)
            ax2.set_ylim(-max_a - 0.5, max_a + 0.5)
            fig.canvas.draw_idle() # actualizamos la pantalla
            plt.pause(0.00001)
            
            print(f"La velocidad de x en el tiempo {t_actual:.2f} segundos es de {vx_suavizado[-1]:.2f} m/s")
            # debe ser vx[-1] ya que debemos hallar la velocidad en ese ultimo punto que acabamos
            # de agregar al vector de centros
            print()
            print(f"La velocidad de y en el tiempo {t_actual:.2f} segundos es de {vy_suavizado[-1]:.2f} m/s")
            print()
            print(f"La aceleracion de x en el tiempo {t_actual:.2f} segundos es de {ax_suavizado[-1]:.2f} m/s^2")
            print()
            print(f"La aceleracion de y en el tiempo {t_actual:.2f} segundos es de {ay_suavizado[-1]:.2f} m/s^2")
            print()
            print(f"La magnitud de la velocidad en el tiempo {t_actual:.2f} segundos es de {mv[-1]:.2f}")
            print()
            print(f"La magnitud de la aceleracion en el tiempo {t_actual:.2f} segundos es de {ma[-1]:.2f}")
            print()
    
            # para dibujar el vector de la velocidad
            theta_v = np.arctan2(vy_suavizado, vx_suavizado) # calcula theta = la tangente inversa de la velocidad en x sobre la velocidad en y
            dvx = np.cos(theta_v) # encontramos el desplazamiento del vector de velocidad en x
            dvy = np.sin(theta_v) # encontramos el desplazamiento del vector de velocidad en y

            # para dibujar el vector de la aceleracion
            theta_a = np.arctan2(ay_suavizado, ax_suavizado) # calcula theta = la tangente inversa de la velocidad en x sobre la velocidad en y
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
        cv2.arrowedLine(mask_vis,(int(xc),int(yc)), (int(xc+150*mv[-1]*dvx[-1]),int(yc+150*mv[-1]*dvy[-1])), (255,0,0), thickness=2,line_type=8)
        cv2.putText(mask_vis, f"Velocidad tangencial {vt[-1]:.2f} m/s", (10,400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.arrowedLine(mask_vis,(int(xc),int(yc)), (int(xc+20*ma[-1]*dax[-1]),int(yc+20*ma[-1]*day[-1])), (0, 0,151), thickness=2,line_type=8)
        cv2.putText(mask_vis, f"Aceleracion tangencial {at[-1]:.2f} m/s^2", (10,450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow('Video', mask_vis)
    else:
        mask_vis = cv2.cvtColor(mask_erode, cv2.COLOR_GRAY2BGR) # para que se pueda mostrar el texto (3 canales)
        cv2.circle(mask_vis, (int(xc),int(yc)), 3, (0,0,255), -1)
        cv2.putText(mask_vis, f"Centro de masa: ({xc:.0f}, {yc:.0f})", (10,350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # para poder mostrar el vector de la velocidad y aceleracion, debemos asegurarnos de que existan almenos
        # 3 posiciones distintas captadas del centro de masa, para poder hallar la primera y segunda derivada
        # que serian la velocidad y aceleracion respectivamente y asi poder mostrar sus magnitudes
        if len(centros_x) >= 2 and len(centros_y) >= 3:
            cv2.arrowedLine(mask_vis,(int(xc),int(yc)), (int(xc+150*mv[-1]*dvx[-1]),int(yc+150*mv[-1]*dvy[-1])), (255,0,0), thickness=2,line_type=8)
            cv2.putText(mask_vis, f"Velocidad tangencial {vt[-1]:.2f} m/s", (10,400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.arrowedLine(mask_vis,(int(xc),int(yc)), (int(xc+20*ma[-1]*dax[-1]),int(yc+20*ma[-1]*day[-1])), (0, 0,151), thickness=2,line_type=8)
            cv2.putText(mask_vis, f"Aceleracion tangencial {at[-1]:.2f} m/s^2", (10,450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow('Video', mask_vis)
    
    key = cv2.waitKey(33) & 0xFF
    if key == 27:  # Presiona la tecla 'Esc' para salir
        break
    
cap.release()
cv2.destroyAllWindows()