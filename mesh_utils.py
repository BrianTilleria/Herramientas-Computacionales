from dolfin import Mesh, MeshFunction

def load_mesh(base_name):
    """
    Esta función carga una malla FEniCS desde archivos .xml generados por dolfin-convert. Se espera que los archivos
    tengan el mismo nombre base y las extensiones adecuadas para la malla, subdominios y fronteras.

    Inputs:
    - base_name: nombre base de los archivos de malla (sin extensión).

    Outputs:
    - mesh: objeto de malla de FEniCS (Mesh).
    - subdomains: objeto MeshFunction que define las subregiones de la malla.
    - boundaries: objeto MeshFunction que define las fronteras de la malla.
    """
    mesh = Mesh(f"{base_name}.xml")
    subdomains = MeshFunction("size_t", mesh, f"{base_name}_physical_region.xml")
    boundaries = MeshFunction("size_t", mesh, f"{base_name}_facet_region.xml")
    return mesh, subdomains, boundaries

def save_mesh_data(mesh, filename_prefix=""):
    """
    Guarda las coordenadas y los elementos de la malla en archivos de texto.
    """
    import numpy as np
    xyz = mesh.coordinates()
    np.savetxt(f"{filename_prefix}coordenadas.txt", xyz, fmt="%.6f", delimiter=" ", header="x y z")
    np.savetxt(f"{filename_prefix}elementos.txt", mesh.cells(), fmt="%d", delimiter=" ", header="Elemento nodo1 nodo2 nodo3")

