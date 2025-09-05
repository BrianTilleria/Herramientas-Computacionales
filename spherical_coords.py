import numpy as np

def appendSpherical_np(xyz):
    """
    Esta función convierte coordenadas cartesianas (x, y, z) a coordenadas esféricas (r, theta, phi)
    y las agrega a un array numpy. La salida es un array de dimensión (M, 6) donde las primeras tres columnas
    son (x, y, z) y las últimas tres columnas son (r, theta, phi).

    Inputs:
    - xyz: array de dimensión (M, 3) con las coordenadas cartesianas.

    Outputs:
    - coords_spherical: array de dimensión (M, 6) con las coordenadas cartesianas y esféricas.

    """
    # Extraer coordenadas cartesianas
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # Calcular r y rho
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)

    # Calcular theta
    theta = np.zeros_like(r)
    mask_z_pos = z > 0
    mask_z_zero = z == 0
    mask_z_neg = z < 0

    theta[mask_z_pos] = np.arctan(rho[mask_z_pos] / z[mask_z_pos])
    theta[mask_z_zero] = np.pi / 2
    theta[mask_z_neg] = np.pi + np.arctan(rho[mask_z_neg] / z[mask_z_neg])

    # Calcular phi
    phi = np.zeros_like(r)

    mask_q1 = (x > 0) & (y >= 0)
    phi[mask_q1] = np.arctan(y[mask_q1] / x[mask_q1])

    mask_q4 = (x > 0) & (y < 0)
    phi[mask_q4] = 2 * np.pi + np.arctan(y[mask_q4] / x[mask_q4])

    mask_x0 = (x == 0)
    phi[mask_x0] = (np.pi / 2) * np.sign(y[mask_x0])

    mask_x_neg = x < 0
    phi[mask_x_neg] = np.pi + np.arctan(y[mask_x_neg] / x[mask_x_neg])

    phi = np.mod(phi, 2 * np.pi)

    return np.column_stack((x, y, z, r, theta, phi))
