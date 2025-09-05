def compute_analytic_E_from_mesh(mesh, params, stim_list,
                                 degree=1,
                                 save=False, save_coords=False,
                                 basename_E="E_matrix_analytic", folder=""):
    """
    Calcula el campo eléctrico desde el potencial analítico proyectado con FEniCS,
    para un espacio de grado arbitrario (P1, P2, ...). No retorna phi_matrix.

    Parámetros
    ----------
    mesh : dolfin.Mesh
        Malla FEniCS.
    params : dict
        Parámetros del modelo.
    stim_list : list of dict
        Lista de configuraciones de estimulación.
    degree : int
        Grado del espacio del polinomio (1 para P1, 2 para P2, etc.). Default es 1.
    save : bool
        Si se guarda el archivo de campo eléctrico. Default es False.
    save_coords : bool
        Si se incluyen coordenadas (x, y, z) en el archivo de salida. Default es False.
    basename_E : str
        Nombre base para el archivo de salida.
    folder : str
        Carpeta para guardar archivos.

    Retorna
    -------
    E_matrix : np.ndarray
        Matriz (num_vertices x 3*N_config) del campo eléctrico proyectado.
    """
    from dolfin import FunctionSpace, VectorFunctionSpace, Function, project, grad, vertex_to_dof_map
    import numpy as np
    from analytic_solution import compute_phi_analytic_multi
    import time
    import os

    #  Definir espacio de funciones
    V = FunctionSpace(mesh, "CG", degree)
    # Definir las coordenadas de los nodos de la malla acorde al espacio V
    coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))

    print("Calculando el potencial analítico")
    start = time.time()                                                 # Inicio el tiempo de ejecución
    phi_matrix = compute_phi_analytic_multi(coords, params, stim_list,
                                            return_matrix=True,
                                            save=True)                  # Calculo el potencial analítico en los nodos del espacio V.
                                                                        # Cada columna de phi_matrix es una configuración de estimulación.
    end = time.time()                                                   # Fin del tiempo de ejecución

    print(f"Tiempo de ejecución para calcular los potenciales fue de: {(end - start) / 60:.4f} minutos")

    # Calculo del campo eléctrico E = -grad(phi) para cada configuración de estimulación
    print("Calculando el campo eléctrico")
    start = time.time()
    W = VectorFunctionSpace(mesh, "CG", degree)                         # Espacio vectorial para E
    E_list = []

    for i in range(phi_matrix.shape[1]):
        phi_f = Function(V)                                            # Creo una función en el espacio V
        phi_f.vector().set_local(phi_matrix[:, i])                     # Cargo el potencial en la función de FEniCS
        phi_f.vector().apply("insert")                                 # Asegurar que los valores estén actualizados

        E = project(-grad(phi_f), W, solver_type="gmres", preconditioner_type="ilu")  # Proyecto al subespacio W el campo eléctrico
        
        # Extraer componentes de E
        Ex, Ey, Ez = E.split(deepcopy=True)
        Ex_vals = Ex.compute_vertex_values(mesh)
        Ey_vals = Ey.compute_vertex_values(mesh)
        Ez_vals = Ez.compute_vertex_values(mesh)
        E_array = np.array([Ex_vals, Ey_vals, Ez_vals]).T
        E_list.append(E_array)
    # Concatenar todas las configuraciones de estimulación
    E_matrix = np.hstack(E_list)
    end = time.time()

    print(f"Tiempo de ejecución para calcular los campos eléctricos fue de: {(end - start) / 60:.4f} minutos")
    tam = phi_matrix.shape[0]
    print("La cantidad de nodos evaluados es:", tam)
    
    # === Guardado opcional ===
    if save:
        header = " ".join([f"Ex_{i+1} Ey_{i+1} Ez_{i+1}" for i in range(E_matrix.shape[1] // 3)])
        filename_E = os.path.join(folder, f"{basename_E}.txt")
        np.savetxt(filename_E, E_matrix, fmt="%.6f", delimiter=" ", header=header)
        print(f"[INFO] Guardado: {filename_E}")

        header_phi = " ".join([f"phi_{i+1}" for i in range(phi_matrix.shape[1])])
        filename_phi = os.path.join(folder, "phi_matrix_analytic.txt")
        np.savetxt(filename_phi, phi_matrix, fmt="%.6f", delimiter=" ", header=header_phi)
        print(f"[INFO] Guardado: {filename_phi}")

    return E_matrix


