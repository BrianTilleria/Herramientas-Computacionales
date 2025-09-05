import os
import numpy as np
from Error import mask_elip, error_calculation, calcular_errores_modulaciones

# Al correr este script, se calcularán los errores para diferentes modulaciones y se mostrarán los resultados en un gráfico si desea. 

# Buscar carpeta de resultados
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results_malla_2M")

# Cargar coordenadas
xyz = np.loadtxt(os.path.join(results_dir, "coordenadas.txt"))

# Calcular errores para diferentes modulaciones
errores = calcular_errores_modulaciones(xyz, 
                                        carpeta=results_dir, 
                                        mostrar_plot=True)


