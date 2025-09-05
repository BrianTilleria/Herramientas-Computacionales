import numpy as np
from numpy.polynomial import legendre as LG
from spherical_coords import appendSpherical_np

def compute_phi_analytic(xyz, params, stim_cfg):
    """
    Esta función calcula el potencial eléctrico en los puntos de la malla de un modelo de 
    cabeza esférico de 3 capas. 
    Inputs:
    - xyz: array de dimensión (M, 3) con las coordenadas cartesianas de los puntos donde se calcula el potencial.
    - params: diccionario con los parámetros del modelo, que incluye:
        - R1, R2, R3: radios de las capas (cerebro, liquido cefalorraquídeo, cráneo).
        - sigma_1: conductividad de la capa.
        - K: factor de atenuación de la conductividad del cráneo.
        - L: factor de aumento de la conductividad del líquido cefalorraquídeo.
        - N: número de términos en la serie de Legendre.
    - stim_cfg: diccionario con la configuración de estimulación, que incluye:
        - source: array de dimensión (3,) con las coordenadas del electrodo fuente.
        - sink: array de dimensión (3,) con las coordenadas del electrodo sumidero.
        - I: corriente inyectada (en Amperios).
    
    Output:
    - phi: array de dimensión (M,) con el potencial eléctrico en cada punto de la malla [V].
    
    """

    R1, R2, R3 = params['R1'], params['R2'], params['R3']               # Radios de las capas
    sigma1 = params['sigma1']                                           # Conductividad de la capa 1 [S/m] 
    sigma2 = sigma1* params['L']                                        # Conductividad de la capa 2 [S/m]
    sigma3 = sigma1 / params['K']                                       # Conductividad de la capa 3 [S/m]
    N = params['N']                                                     # Número de términos en la serie de Legendre

    s21 = sigma2 / sigma1                                               # Ratios de conductividades
    s32 = sigma3 / sigma2                                               
    R32 = R3 / R2                                                       # Ratios de radios
    K1 = stim_cfg['I'] / (4 * np.pi * sigma1)                           # Constante de proporcionalidad

    a = stim_cfg['sink']                                                # Coordenadas de extracción de corriente (sumidero)
    b = stim_cfg['source']                                              # Coordenadas de inyección de corriente (fuente)
    ra = np.linalg.norm(a)                                              # Distancia del sumidero al origen
    rb = np.linalg.norm(b)                                              # Distancia de fuente al origen

    sph_coor = appendSpherical_np(xyz)[:, 3:]                           # Coordenadas esféricas de los puntos de la malla
    r = sph_coor[:, 0]                                                  # Radio de cada punto

    ra_hat = a / np.linalg.norm(a)                                      # Vectores unitarios de a y b
    rb_hat = b / np.linalg.norm(b)                                      
    alpha  = np.arccos(np.dot(ra_hat, rb_hat))                          # Ángulo entre a y b

    r_hat  = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)           # Vectores unitarios de los puntos de la malla
    cos_theta = np.dot(r_hat, rb_hat)                                   # Coseno del ángulo entre r y b
    cos_theta= np.nan_to_num(cos_theta)                                 # Evitar NaNs por división por cero
    sin_theta = np.sqrt(1- cos_theta**2)                                # Seno del ángulo entre r y b
    #cos_beta = sin_theta * cos_phi * np.sin(alpha) + cos_theta * np.cos(alpha)
    cos_beta= np.dot(r_hat,ra_hat)                                      # Coseno del ángulo entre r y a 
    cos_beta= np.nan_to_num(cos_beta)                                   # Evitar NaNs por división por cero        

    # Máscaras para las tres regiones
    rb_unb = np.sqrt(r**2 + rb**2 - 2 * r * rb * cos_theta)
    ra_unb = np.sqrt(r**2 + ra**2 - 2 * r * ra * cos_beta)
    # Potencial en un medio infinito debido a fuente y sumidero
    phi_unbounded = (1 / rb_unb - 1 / ra_unb)
    phi_unbounded= np.nan_to_num(phi_unbounded)
    # Máscaras para las tres regiones
    mask_f1 = r <= R1
    mask_f2 = (R1 < r) & (r <= R2)
    mask_f3 = (R2 < r)

    phi_h_b = np.zeros_like(r)
    phi_h_a = np.zeros_like(r)

    # Cálculo del potencial mediante polinomios de de Legendre. Las constantes se calcularon a mano mediante armónicos esféricos,
    # teniendo en cuenta la condición de Neumann en la capa exterior (no hay corriente saliendo de la cabeza) y continuidad de potencial
    # entre capas. Las expresiones solo contemplan fuente y sumidero dentro de la capa 1 (cerebro).
    for k in range(N):
        l = k + 1
        Pn = LG.Legendre.basis(l)
        L1 = (l + 1) / l
        L2 = 2 * l + 1

        B_str = (l / L2) * (L2 / l - (1 - s32)) + (l / (L2 * L1)) * R32**L2 * (L2 / l - (1 + s32 * L1))
        C_str = (R2**L2 / L2 * (1 - s32) + (R3**L2 / L2) * (1 / L1 + s32)) * l

        # Fuente
        Db = K1 * (L2 / l) * rb**l / (B_str * (1 - s21) * R1**L2 + C_str * (1 + s21 * L1))
        Bb = Db * B_str
        Cb = Db * C_str
        Ab = Bb + Cb * R1**-L2 - K1 * rb**l / R1**L2
        Eb = (1 / L1) * Db * R3**L2

        # Sumidero
        Da = K1 * (L2 / l) * ra**l / (B_str * (1 - s21) * R1**L2 + C_str * (1 + s21 * L1))
        Ba = Da * B_str
        Ca = Da * C_str
        Aa = Ba + Ca * R1**-L2 - K1 * ra**l / R1**L2
        Ea = (1 / L1) * Da * R3**L2

        phi_h_b[mask_f1] += Ab * r[mask_f1]**l * Pn(cos_theta[mask_f1])
        phi_h_b[mask_f2] += (Bb * r[mask_f2]**l + Cb * r[mask_f2]**-(l + 1)) * Pn(cos_theta[mask_f2])
        phi_h_b[mask_f3] += (Db * r[mask_f3]**l + Eb * r[mask_f3]**-(l + 1)) * Pn(cos_theta[mask_f3])

        phi_h_a[mask_f1] += Aa * r[mask_f1]**l * Pn(cos_beta[mask_f1])
        phi_h_a[mask_f2] += (Ba * r[mask_f2]**l + Ca * r[mask_f2]**-(l + 1)) * Pn(cos_beta[mask_f2])
        phi_h_a[mask_f3] += (Da * r[mask_f3]**l + Ea * r[mask_f3]**-(l + 1)) * Pn(cos_beta[mask_f3])

    phi_h_b[mask_f1] = phi_unbounded[mask_f1] * K1 + phi_h_b[mask_f1]
    return phi_h_b - phi_h_a


def compute_phi_analytic_multi(xyz, params, stim_list,
                               return_matrix=True, save=False, save_coords=False,
                               basename="phi_matrix", folder=""):
    
    """
    Esta función calcula el potencial eléctrico en los puntos de la malla de un modelo de 
    cabeza esférico de 3 capas para múltiples configuraciones de estimulación. 

    Inputs:
    - xyz: array de dimensión (M, 3) con las coordenadas cartesianas de los puntos donde se calcula el potencial.
    - params: diccionario con los parámetros del modelo, que incluye:
        - R1, R2, R3: radios de las capas (cerebro, liquido cefalorraquídeo, cráneo).
        - sigma_1: conductividad de la capa.
        - K: factor de atenuación de la conductividad del cráneo.
        - L: factor de aumento de la conductividad del líquido cefalorraquídeo.
        - N: número de términos en la serie de Legendre.
    - stim_list: lista de diccionarios con las configuraciones de estimulación, cada uno que incluye:
        - source: array de dimensión (3,) con las coordenadas del electrodo fuente.
        - sink: array de dimensión (3,) con las coordenadas del electrodo sumidero.
        - I: corriente inyectada (en Amperios).
    - Save: booleano que indica si se deben guardar los resultados en un archivo de texto. Por defecto es False.
    - save_coords: booleano que indica si se deben guardar las coordenadas junto con los potenciales. Por defecto es False.
    - basename: nombre base para el archivo de salida si se guarda. Por defecto es "phi_matrix".
    - folder: carpeta donde se guardará el archivo si se guarda. Por defecto es la carpeta actual.

    Outputs:
    - phi_matrix: array de dimensión (M, S) con el potencial eléctrico en cada punto de la malla para cada configuración de estimulación [V].
      S es el número de configuraciones de estimulación.
    """

    # Calcular el potencial para cada configuración de estimulación
    results = []
    # Recorro cada configuración de estimulación y calculo el potencial con la función compute_phi_analytic
    for idx, stim_cfg in enumerate(stim_list):
        phi = compute_phi_analytic(xyz, params, stim_cfg)
        results.append(phi)                                                # Almaceno el resultado
        print(f"[{idx+1}/{len(stim_list)}] Máx: {np.max(phi):.4e}")        # Imprimo el máximo valor del potencial calculado  

    phi_matrix = np.column_stack(results)                                  # Matriz de potenciales (M, S) 

    # Guardar resultados en un archivo de txt
    if save:
        filename = f"{folder}{basename}.txt"                                    # Nombre del archivo
        header = " ".join([f"phi_{i+1}" for i in range(phi_matrix.shape[1])])   # Cabecera del archivo
        if save_coords:
            data = np.hstack([xyz, phi_matrix])                                 # Incluir coordenadas en el archivo si se desea
            header = "x y z " + header                                          # Cabecera del archivo      
        else:
            data = phi_matrix                                                   # Solo potenciales
        np.savetxt(filename, data, fmt="%.6f", delimiter=" ", header=header)    # Guardar en archivo
        print(f"Guardado: {filename}")                                          # Imprimir mensaje de guardado

    if return_matrix:
        return phi_matrix
