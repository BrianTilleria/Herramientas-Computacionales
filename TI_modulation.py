import numpy as np

def compute_max_modulation(E1_array, E2_array):
	"""
	Esta función calcula la modulación máxima del campo eléctrico resultante de la interferencia temporal
	de dos campos eléctricos E1 y E2, según la fórmula propuesta por Grossman et al. (2017).

	Inputs:
	- E1_array: array de dimensión (M, 3) con las componentes del campo eléctrico E1 en cada punto.
	- E2_array: array de dimensión (M, 3) con las componentes del campo eléctrico E2 en cada punto.

	Output:
	- mod_E_max: array de dimensión (M,) con la modulación máxima del campo eléctrico en cada punto.
	"""
	
	#  Reordeno los vectores para que E1>E2 en norma
	E_1_ti= np.array(np.zeros_like(E1_array))
	E_2_ti= np.array(np.zeros_like(E1_array))
	for ii in range(len(E1_array)):
		if np.linalg.norm(E1_array[ii,:])>np.linalg.norm(E2_array[ii,:]):
			E_1_ti[ii,:]= E1_array[ii,:]
			E_2_ti[ii,:]= E2_array[ii,:]
		else:
			E_2_ti[ii,:]= E1_array[ii,:]
			E_1_ti[ii,:]= E2_array[ii,:]
	
	# Si la proyección de E1 sobre E2 es negativa, roto 180° E1
	proy= np.sum(E_1_ti*E_2_ti, axis=1)
	for ii in range(len(proy)):
		if proy[ii]<0:
			E_1_ti[ii,:]= -E_1_ti[ii,:]

	# Ya tengo organizado E1 y E2 para poder mezclarlos utilizando la fórmula de Grossman

	Gros_cond= np.sqrt(np.sum(E_1_ti*E_2_ti, axis=1)) - np.sqrt(np.sum(E_2_ti*E_2_ti, axis=1))

	# Si Gros_condition es mayor a cero, entonces |proj(E1,E2)|>|E2| sino, cc.
	mod_E_max= np.zeros_like(proy)

	# Calculo la modulación máxima según Grossman et al. (2017)
	for ii in range(len(Gros_cond)):
		if Gros_cond[ii]>0:
			mod_E_max[ii]= 2*np.sqrt(np.dot(E_2_ti[ii,:],E_2_ti[ii,:]))
		else:
			cross_prod= np.cross(E_2_ti[ii,:],(E_1_ti[ii,:]-E_2_ti[ii,:]))
			mod_E_max[ii]= 2* np.linalg.norm(cross_prod)/np.sqrt(np.dot((E_1_ti[ii,:]-E_2_ti[ii,:]),(E_1_ti[ii,:]-E_2_ti[ii,:])))
	
	return mod_E_max
