import os
import numpy as np
from datetime import datetime
from dolfin import *
from spherical_coords import appendSpherical_np
from mesh_utils import load_mesh  
from Nuevo_E_from_Analytic import compute_analytic_E_from_mesh 
from TI_modulation import compute_max_modulation

# === Crear carpeta de resultados ===
mesh_name = "malla_2M"
output_dir = f"results_{mesh_name}"
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Carpeta creada: {output_dir}")

# === Cargar malla ===
mesh, subdomains, boundaries = load_mesh(mesh_name)

# === Guardar coordenadas y conectividad ===
xyz = mesh.coordinates()
np.savetxt(os.path.join(output_dir, "coordenadas.txt"), xyz, fmt="%.6f", delimiter=" ", header="x y z")
np.savetxt(os.path.join(output_dir, "elementos.txt"), mesh.cells(), fmt="%d", delimiter=" ", header="Elemento nodo1 nodo2 nodo3")

# === Parámetros físicos y de simulación ===
params = {
    'R1': 7.8,          # Radio de la capa 1 (cerebro) [cm]
    'R2': 8.0,          # Radio de la capa 2 (líquido cefalorraquídeo) [cm]
    'R3': 8.5,          # Radio de la capa 3 (cráneo) [cm]
    'sigma1': 1 / 300,  # Conductividad del cerebro [S/cm]
    'K': 80,            # Factor de atenuación de la conductividad del cráneo
    'L': 6,             # Factor de aumento de la conductividad del líquido cefalorraquídeo
    'N': 150            # Número de términos en la serie de Legendre
}

# === Configuración de estimulación ===
delta_z = 0.5  # cm

stim_list = [
{'source': np.array([0, 0.5, 6.5+delta_z]), 'sink': np.array([0, 0.5, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, -0.5, 6.5+delta_z]), 'sink': np.array([0, -0.5, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, 1, 6.5+delta_z]), 'sink': np.array([0, 1, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, -1, 6.5+delta_z]), 'sink': np.array([0, -1, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, 1.5, 6.5+delta_z]), 'sink': np.array([0, 1.5, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, -1.5, 6.5+delta_z]), 'sink': np.array([0, -1.5, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, 2, 6.5+delta_z]), 'sink': np.array([0, 2, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, -2, 6.5+delta_z]), 'sink': np.array([0, -2, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, 2.5, 6.5+delta_z]), 'sink': np.array([0, 2.5, 6.5-delta_z]), 'I': 1E-3},
{'source': np.array([0, -2.5, 6.5+delta_z]), 'sink': np.array([0, -2.5, 6.5-delta_z]), 'I': 1E-3},
]

# === Ejecutar cómputo del campo eléctrico analítico ===
degree = 1  # Grado para el subespacio de funciones en el computo del campo eléctrico

E_matrix_analytic = compute_analytic_E_from_mesh(mesh, params, stim_list,
                                        degree=degree,
                                        save=True,
                                        save_coords=False,
                                        basename_E="E_analytic",
                                        folder=output_dir)

print("[INFO] Campo eléctrico analítico calculado y guardado.")

# Separo las componentes del campo eléctrico para cada configuración de estimulación
E1_analytic= E_matrix_analytic[:,0:3]
E2_analytic= E_matrix_analytic[:,3:6]
E3_analytic= E_matrix_analytic[:,6:9]
E4_analytic= E_matrix_analytic[:,9:12]
E5_analytic= E_matrix_analytic[:,12:15]
E6_analytic= E_matrix_analytic[:,15:18]
E7_analytic= E_matrix_analytic[:,18:21]
E8_analytic= E_matrix_analytic[:,21:24]
E9_analytic= E_matrix_analytic[:,24:27]
E10_analytic= E_matrix_analytic[:,27::]

# Computo de las modulaciones máximas para los pares de inyección
print("Computando TI analítico...")
mod_E_analytic_12 = compute_max_modulation(E1_analytic, E2_analytic)
mod_E_analytic_34 = compute_max_modulation(E3_analytic, E4_analytic)
mod_E_analytic_56 = compute_max_modulation(E5_analytic, E6_analytic)
mod_E_analytic_78 = compute_max_modulation(E7_analytic, E8_analytic)
mod_E_analytic_910 = compute_max_modulation(E9_analytic, E10_analytic)

# Guardar resultados de modulaciones en archivos de texto
print("Guardando modulaciones para distintos pares de inyección")
np.savetxt(os.path.join(output_dir, "mod_E_analytic_12.txt"), mod_E_analytic_12, fmt="%.6f", header="Amplitud de TI")
np.savetxt(os.path.join(output_dir, "mod_E_analytic_34.txt"), mod_E_analytic_34, fmt="%.6f", header="Amplitud de TI")
np.savetxt(os.path.join(output_dir, "mod_E_analytic_56.txt"), mod_E_analytic_56, fmt="%.6f", header="Amplitud de TI")
np.savetxt(os.path.join(output_dir, "mod_E_analytic_78.txt"), mod_E_analytic_78, fmt="%.6f", header="Amplitud de TI")
np.savetxt(os.path.join(output_dir, "mod_E_analytic_910.txt"), mod_E_analytic_910, fmt="%.6f", header="Amplitud de TI")



