import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
from datetime import datetime

#-CONTROLLING VARIABLES -----------------------------------------------------------------------------------------------------------

fps = 60            # Frecuencia de actualización de la simulación
stepsPerFrame = 32  # Sub-pasos por fotograma para la precisión del integrador simpléctico
lineTime = 0.5      # Tiempo en segundos de la estela que dejan las masas

M = 100.0   # Masa de la pluma (kg)
L = 10.0    # Longitud de la pluma (m)
m = 1.0    # Masa del gancho (kg)
l = 2.0     # Longitud del cable del gancho (m)
g = 9.81    # Gravedad (m/s^2)
H = 20.0    # Altura inicial del centro de masas de la pluma (m)

# Condiciones iniciales para la pluma y el gancho
xC_0 = 0.0
vxC_0 = 0.0
yC_0 = H
vyC_0 = 0.0

# Gancho (péndulo)
theta_0 = np.radians(0.00)  # Ángulo inicial de desviación del gancho (alpha)
omega_0 = 0.0            # Velocidad angular inicial del gancho

# Pluma (rotación)
phi_0 = np.radians(00.0)    # Inclinación inicial de la pluma respecto a la horizontal
omega_phi_0 = 0.00          # Velocidad angular inicial de rotación de la pluma (rad/s)

#----------------------------------------------------------------------------------------------------------------------------------

# Vector de estado completo: [xC, vxC, yC, vyC, theta, omega_theta, phi, omega_phi]
u_vector = [xC_0, vxC_0, yC_0, vyC_0, theta_0, omega_0, phi_0, omega_phi_0]
deltaT = (1.0 / fps) / float(stepsPerFrame)
u_vectorTimeSnapshots = []

def find_accelerations(xC, vxC, yC, vyC, theta, omega):
    """Calcula las aceleraciones [axC, ayC, alpha_theta] resolviendo el sistema lineal A * acel = B"""
    A = np.array([
        [(M + m),         0,              m * l * np.cos(theta)],
        [0,               (M + m),        m * l * np.sin(theta)],
        [np.cos(theta),   np.sin(theta),  l]
    ])
    
    B = np.array([
        m * l * (omega**2) * np.sin(theta),
        -(M + m) * g - m * l * (omega**2) * np.cos(theta),
        -g * np.sin(theta)
    ])
    
    acel = np.linalg.solve(A, B)
    return acel[0], acel[1], acel[2]

def symplecticEulerOneStep():
    """Integración paso a paso de Euler Simpléctico"""
    xC, vxC, yC, vyC, theta, omega, phi, omega_phi = u_vector
    
    # 1. Obtener aceleraciones instantáneas
    axC, ayC, alpha_theta = find_accelerations(xC, vxC, yC, vyC, theta, omega)
    alpha_phi = 0.0  # Como d(dL/d_dot_phi)/dt = 0 -> d2_phi/dt2 = 0 (aceleración angular de la pluma es nula)
    
    # 2. Actualizar velocidades (Euler Simpléctico)
    u_vector[1] += axC * deltaT
    u_vector[3] += ayC * deltaT
    u_vector[5] += alpha_theta * deltaT
    u_vector[7] += alpha_phi * deltaT
    
    # 3. Actualizar posiciones con las nuevas velocidades
    u_vector[0] += u_vector[1] * deltaT
    u_vector[2] += u_vector[3] * deltaT
    u_vector[4] += u_vector[5] * deltaT
    u_vector[6] += u_vector[7] * deltaT
    
    u_vectorTimeSnapshots.append(copy.deepcopy(u_vector))

def getCoordinatesForParticle(_particle, _axis):
    """Obtiene la historia de posiciones para la estela trazadora"""
    line = []
    # _particle == 0: Centro de masas de la pluma (xC, yC)
    if _particle == 0:
        idx = 0 if _axis == 0 else 2
        for snap in u_vectorTimeSnapshots:
            line.append(snap[idx])
        return line
    # _particle == 1: Posición del gancho (xm, ym)
    if _particle == 1:
        for snap in u_vectorTimeSnapshots:
            xC_i, yC_i, th_i = snap[0], snap[2], snap[4]
            if _axis == 0:
                line.append(xC_i + l * np.sin(th_i))
            else:
                line.append(yC_i - l * np.cos(th_i))
        return line

# Configuraciones de la ventana gráfica
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-L, L)
ax.set_ylim(-2, H + 2)
ax.grid(True)

# Líneas para la pluma, cable y estelas
line_estela_cm, = ax.plot([], [], lw=1, color='green', linestyle='--', label='Estela CM')
line_estela_gancho, = ax.plot([], [], lw=1.5, color='red', label='Estela Gancho')
bar_pluma, = ax.plot([], [], lw=4, color='black', label='Pluma (M)')
bar_cable, = ax.plot([], [], lw=2, color='blue', linestyle='--', label='Cable')

scats = []
counter = 0

def animation_frame(i):
    global counter, starttime
    counter += 1

    if i == 0:
        starttime = datetime.now()

    # Avanzar los sub-pasos del integrador
    for _ in range(stepsPerFrame):
        symplecticEulerOneStep()

    # Mantener únicamente las posiciones correspondientes a 'lineTime' segundos
    if counter > fps * lineTime:
        for _ in range(stepsPerFrame):
            if len(u_vectorTimeSnapshots) > 0:
                del u_vectorTimeSnapshots[0]

    # Actualizar títulos e información
    elapsed = (datetime.now() - starttime).total_seconds()
    averageFps = round(i / elapsed) if elapsed > 0 else 0
    ax.set_title(f'FPS Medio: {averageFps} || Estela: {lineTime} s\nPluma: {L} m, {M} kg || Gancho: {l} m, {m} kg')

    # Posiciones y ángulos actuales
    xC_curr, yC_curr = u_vector[0], u_vector[2]
    th_curr = u_vector[4]
    phi_curr = u_vector[6]

    xm_curr = xC_curr + l * np.sin(th_curr)
    ym_curr = yC_curr - l * np.cos(th_curr)

    # 1. Dibujar estelas de trayectoria
    line_estela_cm.set_data(getCoordinatesForParticle(0, 0), getCoordinatesForParticle(0, 1))
    line_estela_gancho.set_data(getCoordinatesForParticle(1, 0), getCoordinatesForParticle(1, 1))

    # 2. Dibujar estructura física de la pluma (rotada phi) y el cable
    x_pluma_1 = xC_curr - (L / 2.0) * np.cos(phi_curr)
    x_pluma_2 = xC_curr + (L / 2.0) * np.cos(phi_curr)
    y_pluma_1 = yC_curr - (L / 2.0) * np.sin(phi_curr)
    y_pluma_2 = yC_curr + (L / 2.0) * np.sin(phi_curr)

    bar_pluma.set_data([x_pluma_1, x_pluma_2], [y_pluma_1, y_pluma_2])
    bar_cable.set_data([xC_curr, xm_curr], [yC_curr, ym_curr])

    # 3. Dibujar puntos/masas
    global scats
    for scat in scats:
        scat.remove()
    scats = []

    # Masa del centro de la pluma (verde) y Masa del gancho (roja)
    scats.append(ax.scatter(xC_curr, yC_curr, color='green', s=40, zorder=3))
    scats.append(ax.scatter(xm_curr, ym_curr, color='red', s=80, zorder=3))

ani = animation.FuncAnimation(fig, func=animation_frame, frames=None, interval=1000.0 / fps, cache_frame_data=False)
plt.legend(loc='upper right')
plt.show()