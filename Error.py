import numpy as np
import matplotlib.pyplot as plt

def mask_elip(xyz,tol):
  """
  Esta función genera una máscara booleana para un elipsoide centrado en (0,0,6.5). La máscara es True para los puntos dentro del elipsoide
  y False para los puntos fuera del elipsoide. El elipsoide define la región de interés (ROI) para aplicar interferencia temporal.
  Los errores entre la solución analítica y la solución FEM se calculan solo dentro de esta región.

  Inputs:
  - xyz: array de dimensión (M, 3) con las coordenadas cartesianas.
  - tol: tolerancia para incluir puntos cercanos a la superficie del elipsoide.
  
  Outputs:
  - mask_elipsoide: array booleano de dimensión (M,) que indica si cada punto está dentro del elipsoide.
  """
  # Extraer coordenadas cartesianas
  x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
  # Ecuación del elipsoide (x^2/a^2 + y^2/b^2 + (z-z0)^2/c^2 <= 1)
  elipsoide_3d= x**2/1 + y**2/1 + (z-6.5)**2/np.sqrt(3)
  # Crear máscara booleana
  mask_elipsoide = elipsoide_3d <= 1 +tol

  return mask_elipsoide


def error_calculation(TI_analytic, TI_FEM, elipsoide_mask):
  """
  Esta función calcula el error relativo medio entre dos arrays (TI_analytic y TI_FEM) dentro de una región definida por una máscara 
  booleana (elipsoide_mask). El error se define como la media del valor absoluto de la diferencia entre los dos arrays 
  entre la media del array analítico.

  Inputs:
  - TI_analytic: array de dimensión (M,) con los valores analíticos.
  - TI_FEM: array de dimensión (M,) con los valores obtenidos por el método FEM.
  - elipsoide_mask: array booleano de dimensión (M,) que indica la región de interés (True para puntos dentro del elipsoide).

  Outputs:
  - error: valor escalar que representa el error relativo medio entre los dos arrays dentro de la región definida por la máscara.
  """
  # Aplicar la máscara para considerar solo los puntos dentro del elipsoide
  TI_analytic_elipsoide= TI_analytic[elipsoide_mask]
  TI_FEM_elipsoide= TI_FEM[elipsoide_mask]
  
  # Calcular el error relativo medio
  error_mean= np.mean(np.abs(TI_analytic_elipsoide-TI_FEM_elipsoide))
  error= error_mean/np.mean(TI_analytic_elipsoide)
  return error


def calcular_errores_modulaciones(xyz, tol=1e-3, mostrar_plot=True, carpeta="./", guardar=True):
    """
    Esta función calcula los errores relativos entre las modulaciones analíticas y las modulaciones obtenidas por el método FEM
    para cinco pares de electrodos en diferentes posiciones.
    Los errores se calculan solo para los puntos dentro del elipsoide definido por la función mask_elip.

    Inputs:
    - xyz : ndarray
        Array de dimensión (M, 3) con las coordenadas cartesianas.
    - tol : float, optional
        Tolerancia para incluir puntos cercanos a la superficie del elipsoide. Por defecto es 1e-3.
    - mostrar_plot : bool, optional
        Si es True, se muestra un gráfico de los errores. Por defecto es True.
    - carpeta : str, optional
        Ruta de la carpeta donde se encuentran los archivos de las modulaciones y donde se guardará el archivo de errores. 
        Por defecto es "./". (ASEGURARSE DE QUE SE TENGAN TODAS LAS MODULACIONES MÁXIMAS EN ESTA CARPETA).
    - guardar : bool, optional
    
    Outputs:
    error_vector : ndarray
        Vector con los errores relativos.

    """
  

    # Crear la máscara del elipsoide
    elipsoide_mask = mask_elip(xyz, tol)

    
    pares = ["12", "34", "56", "78", "910"]   # Nombres de los archivos para cada par de electrodos

    # Inicializar el vector de errores
    error_vector = np.zeros(len(pares))

    for i, par in enumerate(pares):
        mod_analytic = np.loadtxt(f"{carpeta}/mod_E_analytic_{par}.txt")              # Cargar la modulación analítica
        mod_fem = np.loadtxt(f"{carpeta}/mod_E_FEM_{par}.txt")                        # Cargar la modulación obtenida por FEM
        error_vector[i] = error_calculation(mod_analytic, mod_fem, elipsoide_mask)    # Calcular el error y almacenarlo en el vector

    if guardar:
        np.savetxt(f"{carpeta}/error_vector.txt", error_vector, fmt="%.6f",
                   header="Error para cada par de inyección")                          # Guardar el vector de errores en un archivo

    # Plotear los errores si se solicita
    if mostrar_plot:
        x = np.arange(1, len(pares) + 1)                                               
        plt.figure(figsize=(6, 4))
        plt.plot(x, error_vector*100, marker='o', linestyle='-', color='royalblue',
                 linewidth=2, markersize=8, label='Error')
        plt.title("Error para cada par de inyección", fontsize=14)
        plt.xlabel("Distancia entre pares de inyección [cm]", fontsize=12)
        plt.ylabel("Error [%]", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.show()