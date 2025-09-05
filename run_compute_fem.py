from mesh_utils import load_mesh, save_mesh_data
from spherical_coords import appendSpherical_np
from compute_FEM import compute_phi_fem_multi
from TI_modulation import compute_max_modulation
import os
from datetime import datetime
import numpy as np
import time


def main():
	"""
	Función principal para ejecutar la simulación de potencial eléctrico y campo eléctrico en un modelo de cabeza esférico de 3 capas.
	Configura los parámetros físicos, la malla, las configuraciones de estimulación, y guarda los resultados en una carpeta con timestamp.
	""" 
	# === Crear carpeta de resultados ===
	mesh_name = "malla_2M"		                            # Nombre de la malla a cargar (sin extensión)
	output_dir = f"results_{mesh_name}"			# Nombre de la carpeta de resultados
	os.makedirs(output_dir, exist_ok=True)					# Crear la carpeta si no existe
	print(f"[INFO] Carpeta creada: {output_dir}")			# Imprimir mensaje de creación

	# === Cargar malla ===
	mesh, subdomains, boundaries = load_mesh(mesh_name)     # La malla fue creada en Gmsh y convertida a .xdmf con dolfin-convert.

	# === Coordenadas cartesianas y esféricas ===
	xyz = mesh.coordinates()									
	spherical = appendSpherical_np(xyz)     		  		# Agrego coordenadas esféricas (r, theta, phi). Función en spherical_coords.py

	# Guardar coordenadas y elementos de la malla
	np.savetxt(os.path.join(output_dir, "coordenadas.txt"), xyz, fmt="%.6f", delimiter=" ", header="x y z")
	np.savetxt(os.path.join(output_dir, "elementos.txt"), mesh.cells(), fmt="%d", delimiter=" ", header="Elemento nodo1 nodo2 nodo3")

	# === Parámetros físicos y de simulación === Los valores son de la literatura.
	params = {
	'R1': 7.8,
	'R2': 8,
	'R3': 8.5,
	'sigma1': 1/300,
	'K': 80,
	'L': 6,
	'N': 85
	}

	# === Configuración de estimulación ===
	# Simulo para todos los pares de inyección 1cm-3cm-5cm


	delta_z= 1/2     # Separación entre electrodos

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

	# Computo el tiempo que demora calcular los potenciales en FEM
	start = time.time()									# Inicio el tiempo de ejecución

	# === Simulación FEM === Calculo potencial y campo eléctrico para todas las configuraciones de estimulación.

	phi_matrix_fem, E_matrix_fem = compute_phi_fem_multi(
	mesh, subdomains, boundaries, stim_list, params,
	return_matrix=True, save=False,
	basename=os.path.join(output_dir, "phi_matrix_fem"),basename2=os.path.join(output_dir, "E_matrix_fem")
	)  						# Calculo FEM y almaceno resultados en la carpeta de resultados creada.
	end = time.time()
	elapsed_minutes = (end - start) / 60 				# Tiempo en minutos

	# Imprimir tiempo de ejecución
	print(f"Tiempo de ejecución para calcular los potenciales y campos eléctricos FEM es de: {elapsed_minutes:.4f} minutos")
	# Guardar potenciales y campo eléctrico de resultados en archivos separados
	print("Almacenando Phi_matrix_FEM y E_matrix_FEM...")
	np.savetxt(os.path.join(output_dir, "phi_matrix_FEM.txt"), phi_matrix_fem, fmt="%.6f", delimiter=" ", header="x y z")
	np.savetxt(os.path.join(output_dir, "E_matrix_FEM.txt"), E_matrix_fem, fmt="%.6f", delimiter=" ", header="Ex Ey Ez")
	print("Almacenado correctamente")


	# === Cálculo de interferencia temporal ===
	print("Computando interferencia temporal...")


	# Cada par de inyección tiene 3 columnas (Ex, Ey, Ez) en E_matrix_fem que contiene los 10 pares de inyección.

	E1_FEM= E_matrix_fem[:,0:3]
	E2_FEM= E_matrix_fem[:,3:6]
	E3_FEM= E_matrix_fem[:,6:9]
	E4_FEM= E_matrix_fem[:,9:12]
	E5_FEM= E_matrix_fem[:,12:15]
	E6_FEM= E_matrix_fem[:,15:18]
	E7_FEM= E_matrix_fem[:,18:21]
	E8_FEM= E_matrix_fem[:,21:24]
	E9_FEM= E_matrix_fem[:,24:27]
	E10_FEM= E_matrix_fem[:,27::]

	# Computo la modulación de la interferencia temporal para cada par de inyección
	mod_E_fem_12 = compute_max_modulation(E1_FEM, E2_FEM)
	mod_E_fem_34 = compute_max_modulation(E3_FEM, E4_FEM)
	mod_E_fem_56 = compute_max_modulation(E5_FEM, E6_FEM)
	mod_E_fem_78 = compute_max_modulation(E7_FEM, E8_FEM)
	mod_E_fem_910 = compute_max_modulation(E9_FEM, E10_FEM)

	# Guardar resultados de interferencia temporal
	print("Guardando interferencia temporal de todos los pares de inyección...")

	np.savetxt(os.path.join(output_dir, "mod_E_FEM_12.txt"), mod_E_fem_12, fmt="%.6f", header="Amplitud de TI para FEM")
	np.savetxt(os.path.join(output_dir, "mod_E_FEM_34.txt"), mod_E_fem_34, fmt="%.6f", header="Amplitud de TI para FEM")
	np.savetxt(os.path.join(output_dir, "mod_E_FEM_56.txt"), mod_E_fem_56, fmt="%.6f", header="Amplitud de TI para FEM")
	np.savetxt(os.path.join(output_dir, "mod_E_FEM_78.txt"), mod_E_fem_78, fmt="%.6f", header="Amplitud de TI para FEM")
	np.savetxt(os.path.join(output_dir, "mod_E_FEM_910.txt"), mod_E_fem_910, fmt="%.6f", header="Amplitud de TI para FEM")

if __name__ == "__main__":
    main()

