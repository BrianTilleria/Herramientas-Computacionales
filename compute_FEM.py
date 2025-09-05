from dolfin import *
import numpy as np

def compute_phi_fem_multi(mesh, subdomains, boundary_markers, stim_list, params,
                          return_matrix=True, save=False, basename="phi_matrix_fem", basename2= "E_matrix_fem", folder=""):
    
    """
    Esta función calcula el potencial eléctrico y el campo eléctrico en los puntos de la malla de un modelo de cabeza de 3 capas 
    usando el método de elementos finitos (FEM). Se pueden simular múltiples configuraciones de estimulación. El grado de los polinomios 
    utilizados puede ajustarse según la precisión deseada (por coste computacional se definen polinomios de Lagrange de grado 1).

    Inputs:
    - mesh: objeto de malla de FEniCS (Mesh) que define la geometría del modelo de cabeza de 3 capas. Formato .xdmf convertido desde Gmsh.
    - subdomains: objeto MeshFunction que define las subregiones de la malla (cerebro, LCR, cráneo). 
    - boundary_markers: objeto MeshFunction que define las fronteras de la malla.
    - stim_list: lista de diccionarios, cada uno con la configuración de estimulación:
        - source: array de dimensión (3,) con las coordenadas del electrodo fuente.
        - sink: array de dimensión (3,) con las coordenadas del electrodo sumidero.
        - I: corriente inyectada (en Amperios).
    - params: diccionario con los parámetros del modelo, que incluye:
        - sigma1: conductividad de la capa (cerebro).
        - K: factor de atenuación de la conductividad del cráneo.
        - L: factor de aumento de la conductividad del líquido cefalorraquídeo.
    - save: booleano que indica si se deben guardar las matrices de potencial y campo eléctrico en archivos de texto. Por defecto es False.
    - basename: nombre base para los archivos de salida si se guardan. Por defecto es "phi_matrix_fem" para potencial
    - basename2: nombre base para los archivos de salida si se guardan. Por defecto es "E_matrix_fem" para campo eléctrico
    - folder: carpeta donde se guardarán los archivos si se guardan. Por defecto es la carpeta actual.

    Outputs:
    - phi_matrix: array de dimensión (M, S) con el potencial eléctrico en cada punto de la malla para cada configuración de estimulación [V].
      M es el número de nodos en la malla y S es el número de configuraciones de estimulación.
    - E_matrix: array de dimensión (M, S*3) con las componentes del campo eléctrico en cada punto de la malla para cada configuración 
    de estimulación [V/m]. M es el número de nodos en la malla y S es el número de configuraciones de estimulación. 

    """
    # Parámetros físicos
    sig_brain = params['sigma1']
    K = params['K']
    sig_CSF = sig_brain * params['L']
    sig_skull = sig_brain / K

    # Definir números de subdominios (Definidos en Gmsh)
    brainvol = 32
    csfvol   = 64
    skullvol = 96

    # Definimos las funciones de prueba y ensayo
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    dx_sub = dx(subdomain_data=subdomains)
    # Definimos el problema variacional con la formulación débil. 
    a = (
        inner(Constant(sig_brain) * grad(u), grad(v)) * dx_sub(brainvol) +
        inner(Constant(sig_CSF)   * grad(u), grad(v)) * dx_sub(csfvol) +
        inner(Constant(sig_skull) * grad(u), grad(v)) * dx_sub(skullvol)
    )

    # Condiciones de frontera de Neumann están implícitas en la formulación débil. 
    # Hacemos el ensamble y preparamos el solver.
    A = assemble(a)
    solver = KrylovSolver("cg", "ilu")
    solver.parameters["maximum_iterations"] = 1000
    solver.parameters["relative_tolerance"] = 1e-8
    solver.parameters["absolute_tolerance"] = 0.0
    solver.parameters["monitor_convergence"] = True

    # Almacenar resultados
    phi_matrix = []
    E_matrix= []	
    # Recorro cada configuración de estimulación y calculo el potencial
    for idx, stim in enumerate(stim_list):
        print(f"Resolviendo configuración {idx+1}/{len(stim_list)}...")
        b = assemble(Constant(0) * v * dx)
        PointSource(V, Point(*stim['source']),  stim['I']).apply(b)   # Fuente
        PointSource(V, Point(*stim['sink']), -stim['I']).apply(b)     # Sumidero
        phi = Function(V)                                             
        solver.solve(A, phi.vector(), b)                              # Resuelvo el sistema lineal
	# Extraer potencial en nodos
        phi_vec = phi.compute_vertex_values(mesh)
        phi_matrix.append(phi_vec)

        V = phi.function_space()       # Creo un espacio con las mismas cualidades de phi
        mesh = V.mesh()                # Tomo el mismo mallado
        degree = V.ufl_element().degree()  # El mismo elemento
        W = VectorFunctionSpace(mesh, 'CG', degree)  # Utilizo polinomio de Lagrange del mismo grado que V
        
        # Calculo el campo eléctrico E = -grad(phi)
        E = Function(W)
        k = Constant(1.0)
        E = project(-k * grad(phi), W, solver_type="gmres", preconditioner_type="ilu")   # Proyecto al subespacio W
        
        # Obtener componentes de E 
        Ex, Ey, Ez = E.split(deepcopy=True)
        Ex_values = Ex.compute_vertex_values(mesh)
        Ey_values = Ey.compute_vertex_values(mesh)
        Ez_values = Ez.compute_vertex_values(mesh)

        # Guardar campo eléctrico
        E_array = np.array([Ex_values, Ey_values, Ez_values]).T
        E_matrix.append(E_array)


    # Reorganizar a matriz (nodos x configuraciones)
    phi_matrix = np.column_stack(phi_matrix)
    E_matrix = np.column_stack(E_matrix)

    # Guardar resultados en archivos de texto si se desea
    if save:
        filename = f"{folder}{basename}.txt"
        np.savetxt(filename, phi_matrix, fmt="%.6f", delimiter=" ",
                   header=" ".join([f"phi_{i+1}" for i in range(phi_matrix.shape[1])])) # Guardar en archivo 
        print(f"Guardado: {filename}")

        filename = f"{folder}{basename2}.txt"
        np.savetxt(filename, E_matrix, fmt="%.6f", delimiter=" ",
                   header=" ".join([f"E_{i+1}" for i in range(E_matrix.shape[1])]))     # Guardar en archivo 
        print(f"Guardado: {filename}")



    if return_matrix:
        return phi_matrix, E_matrix



