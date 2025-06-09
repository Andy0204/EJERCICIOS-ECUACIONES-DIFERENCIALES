import numpy as np
import matplotlib.pyplot as plt

# ======================
# MÉTODOS NUMÉRICOS
# ======================

def heun(f, t0, y0, h, n):
    t, y = [t0], [y0]
    for _ in range(n):
        t_next = t[-1] + h
        y_pred = y[-1] + h * f(t[-1], y[-1])
        y_next = y[-1] + h/2 * (f(t[-1], y[-1]) + f(t_next, y_pred))
        t.append(t_next)
        y.append(y_next)
    return np.array(t), np.array(y)

def rk4(f, t0, y0, h, n):
    t, y = [t0], [y0]
    for _ in range(n):
        ti, yi = t[-1], y[-1]
        k1 = h * f(ti, yi)
        k2 = h * f(ti + h/2, yi + k1/2)
        k3 = h * f(ti + h/2, yi + k2/2)
        k4 = h * f(ti + h, yi + k3)
        y.append(yi + (k1 + 2*k2 + 2*k3 + k4)/6)
        t.append(ti + h)
    return np.array(t), np.array(y)

# ======================
# PROBLEMA 1: CRECIMIENTO POBLACIONAL
# ======================
def problema_poblacional():
    k, NM = 0.000095, 5000
    f = lambda t, N: k * N * (NM - N)
    t0, N0, h, n = 0, 100, 0.6667, 30
    t_heun, N_heun = heun(f, t0, N0, h, n)
    t_rk4, N_rk4 = rk4(f, t0, N0, h, n)

    plt.figure(figsize=(7, 5))
    plt.plot(t_heun, N_heun, 'o-', label="Heun")
    plt.plot(t_rk4, N_rk4, 's--', label="Runge-Kutta 4")
    plt.title("Problema 1: Crecimiento Poblacional")
    plt.xlabel("Tiempo")
    plt.ylabel("Población")
    plt.legend()
    plt.grid(True)
    plt.savefig("poblacion.png")
    plt.show()

# ======================
# PROBLEMA 2: CRECIMIENTO TUMORAL
# ======================
def problema_tumoral():
    alpha, k, nu = 0.8, 60, 0.25
    f = lambda t, A: alpha * A * (1 - (A/k)**nu)
    t0, A0, h, n = 0, 1, 1, 30
    t_heun, A_heun = heun(f, t0, A0, h, n)
    t_rk4, A_rk4 = rk4(f, t0, A0, h, n)

    plt.figure(figsize=(7, 5))
    plt.plot(t_heun, A_heun, 'o-', label="Heun")
    plt.plot(t_rk4, A_rk4, 's--', label="Runge-Kutta 4")
    plt.title("Problema 2: Crecimiento Tumoral")
    plt.xlabel("Tiempo")
    plt.ylabel("Área del tumor")
    plt.legend()
    plt.grid(True)
    plt.savefig("tumor.png")
    plt.show()

# ======================
# PROBLEMA 3: CAÍDA LIBRE CON FRICCIÓN
# ======================
def problema_caida():
    m, g, k = 5, 9.81, 0.05
    f = lambda t, v: -g + (k/m) * v**2
    t0, v0, h, n = 0, 0, 0.5, 30
    t_heun, v_heun = heun(f, t0, v0, h, n)
    t_rk4, v_rk4 = rk4(f, t0, v0, h, n)

    plt.figure(figsize=(7, 5))
    plt.plot(t_heun, v_heun, 'o-', label="Heun")
    plt.plot(t_rk4, v_rk4, 's--', label="Runge-Kutta 4")
    plt.title("Problema 3: Caída Libre con Fricción Cuadrática")
    plt.xlabel("Tiempo")
    plt.ylabel("Velocidad")
    plt.legend()
    plt.grid(True)
    plt.savefig("caida.png")
    plt.show()

# ======================
# EJECUCIÓN PRINCIPAL
# ======================
if __name__ == "__main__":
    problema_poblacional()
    problema_tumoral()
    problema_caida()
    print("✔ Gráficos generados y guardados como 'poblacion.png', 'tumor.png' y 'caida.png'.")
