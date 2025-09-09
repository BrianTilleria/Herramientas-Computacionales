import os
import numpy as np
import iso2mesh as i2m
import matplotlib.pyplot as plt
import iso2mesh as i2m
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.tri as mtri

# Buscar carpeta de resultados
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results_malla_2M")

## Cargar datos
coords= np.loadtxt(os.path.join(results_dir, "coordenadas.txt"))
phi_values_analytic = np.loadtxt(os.path.join(results_dir, "phi_matrix_analytic.txt"))
phi_values_FEM = np.loadtxt(os.path.join(results_dir, "phi_matrix_FEM.txt"))
cells = np.loadtxt(os.path.join(results_dir, "elementos.txt"))
cells1 = cells.astype(int) + 1
mod_E_FEM_12= np.loadtxt(os.path.join(results_dir, "mod_E_FEM_12.txt"))
mod_E_analytic_12= np.loadtxt(os.path.join(results_dir, "mod_E_analytic_12.txt"))

# Valors máximos y mínmos de potencial
vmin, vmax = 0.0, float(mod_E_FEM_12.max())

### Prueba de CUTMESHCUT

# cut the mesh and interpolate the nodal-values at nodalval=10
cutpos, cutval, facedata, _, _ = i2m.qmeshcut(cells1, coords, mod_E_analytic_12, 'x=0')
i2m.plotmesh(np.column_stack([cutpos, cutval]), facedata.tolist(), 'facecolor','interp', linewidth=0.1)
#i2m.plotmesh( coords , i2m.meshedge(facedata[:,:3]), parent=hh, linewidth=0.2, cmap= 'jet')
ax = plt.gca()
ax.view_init(elev=0, azim=0)
# Colorbar con etiqueta en V/m
norm = Normalize(vmin=vmin, vmax=vmax)
mappable = cm.ScalarMappable(norm=norm, cmap='jet')
mappable.set_array([])  # requerido por Matplotlib
cb = plt.colorbar(mappable, ax=ax, pad=0.06, shrink=0.85)
cb.set_label('TI [V/m]', fontsize=12)

plt.tight_layout()
ax.set_title('TI - FEM', fontsize=14, pad=12)
ax.set_ylabel('y [cm]', fontsize=12, labelpad=8)
ax.set_zlabel('z [cm]', fontsize=12, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.savefig(os.path.join(results_dir, "TI_FEM_cm_cut.png"), dpi=600)
print("Figura guardada en 'results_malla_2M/TI_FEM_cm_cut.png'")
plt.show()


# Graficamos el potencial FEM
colmap = plt.get_cmap("jet")
# Superficie coloreada por potencial
i2m.plotmesh(np.column_stack([coords, mod_E_FEM_12]),cells1,
             '(x<0) & (x>-0.3)', 'edgecolor', 'k',cmap= colmap, linewidth=0.1)
ax = plt.gca()
ax.view_init(elev=0, azim=0)
ax.set_xlabel('')                          # sin nombre
ax.set_xticks([])                          # sin ticks
ax.tick_params(axis='x', which='both',     # sin marcas ni etiquetas
               labelbottom=False, length=0)

# === Título y ejes con unidades ===
ax.set_title('TI - FEM', fontsize=14, pad=12)
ax.set_ylabel('y [cm]', fontsize=12, labelpad=8)
ax.set_zlabel('z [cm]', fontsize=12, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=10)

# Colorbar con etiqueta en V/m
norm = Normalize(vmin=vmin, vmax=vmax)
mappable = cm.ScalarMappable(norm=norm, cmap=colmap)
mappable.set_array([])  # requerido por Matplotlib
cb = plt.colorbar(mappable, ax=ax, pad=0.06, shrink=0.85)
cb.set_label('TI [V/m]', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "TI_FEM.png"), dpi=600)
print("Figura guardada en 'results_malla_2M/TI_FEM.png'")
plt.show()


# Graficamos la modulation depth analítica
vmin, vmax = 0.0, float(mod_E_analytic_12.max())
colmap = plt.get_cmap("jet")
# Superficie coloreada por potencial
i2m.plotmesh(np.column_stack([coords, mod_E_analytic_12]),cells1,
             '(x<0) & (x>-0.3)', 'edgecolor', 'k', 'facecolor',cmap= colmap, linewidth=0.1)
ax = plt.gca()
ax.view_init(elev=0, azim=0)
ax.set_xlabel('')                          # sin nombre
ax.set_xticks([])                          # sin ticks
ax.tick_params(axis='x', which='both',     # sin marcas ni etiquetas
               labelbottom=False, length=0)

# === Título y ejes con unidades ===
ax.set_title('TI - Analítica', fontsize=14, pad=12)
ax.set_ylabel('y [cm]', fontsize=12, labelpad=8)
ax.set_zlabel('z [cm]', fontsize=12, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=10)

# Colorbar con etiqueta en V/m
norm = Normalize(vmin=vmin, vmax=vmax)
mappable = cm.ScalarMappable(norm=norm, cmap=colmap)
mappable.set_array([])  # requerido por Matplotlib
cb = plt.colorbar(mappable, ax=ax, pad=0.06, shrink=0.85)
cb.set_label('TI [V/m]', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "TI_analytic.png"), dpi=600)
print("Figura guardada en 'results_malla_2M/TI_analytic.png'")
plt.show()

#### Error absoluto entre FEM y Analítico 
error_abs_12 = np.abs(mod_E_FEM_12 - mod_E_analytic_12)
vmin, vmax = 0.0, float(error_abs_12.max())

# Graficamos el error absoluto
colmap = plt.get_cmap("hot")
# Superficie coloreada por potencial
i2m.plotmesh(np.column_stack([coords, error_abs_12]),cells1,'(x<0) & (x>-0.3)', 'facecolor',cmap= colmap, linewidth=0.05)
ax = plt.gca()
ax.view_init(elev=0, azim=0)
ax.set_xlabel('')                          # sin nombre
ax.set_xticks([])                          # sin ticks
ax.tick_params(axis='x', which='both',     # sin marcas ni etiquetas
               labelbottom=False, length=0)

# === Título y ejes con unidades ===
ax.set_title('Error absoluto TI', fontsize=14, pad=12)
ax.set_ylabel('y [cm]', fontsize=12, labelpad=8)
ax.set_zlabel('z [cm]', fontsize=12, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=10)

# Colorbar con etiqueta en V/m
norm = Normalize(vmin=vmin, vmax=vmax)
mappable = cm.ScalarMappable(norm=norm, cmap=colmap)
mappable.set_array([])  # requerido por Matplotlib
cb = plt.colorbar(mappable, ax=ax, pad=0.06, shrink=0.85)
cb.set_label('Error [V/m]', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "error_12.png"), dpi=600)
print("Figura guardada en 'results_malla_2M/error_12.png'")
plt.show()


# Campo eléctrico analítico
E_matrix_analytic = np.loadtxt(os.path.join(results_dir, "E_analytic.txt"))
E_matrix_FEM = np.loadtxt(os.path.join(results_dir, "E_matrix_FEM.txt"))

# Agregamos una mascara para graficar solo en la región x<0 y x>-0.2
mask = (coords[:, 0] > -0.2) & (coords[:, 0] < 0) 
E_mag_analytic = np.linalg.norm(E_matrix_analytic[:, 0:2], axis=1)
E_mag_FEM = np.linalg.norm(E_matrix_FEM[:, 0:2], axis=1)

# Graficamos el campo eléctrico FEM

colmap = plt.get_cmap("jet")
idx= np.where(mask)[0]
i2m.plotmesh(np.column_stack([coords, E_mag_FEM /np.max(E_mag_FEM)]), cells1,'(x<0) & (x>-0.3)',cmap= colmap)
ax = plt.gca()
ax.view_init(elev=0, azim=0)
ax.quiver(coords[idx,0], coords[idx,1], coords[idx,2], E_matrix_FEM[idx,0], E_matrix_FEM[idx,1], E_matrix_FEM[idx,2], 
          length=0.4, normalize=True, cmap=colmap, color='w', linewidth=0.2)
ax.set_xlabel('')                          # sin nombre
ax.set_xticks([])                          # sin ticks
ax.tick_params(axis='x', which='both',     # sin marcas ni etiquetas
               labelbottom=False, length=0)

# === Título y ejes con unidades ===
ax.set_title('Campo eléctrico FEM', fontsize=14, pad=12)
ax.set_ylabel('y [cm]', fontsize=12, labelpad=8)
ax.set_zlabel('z [cm]', fontsize=12, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=10)

# Colorbar con etiqueta en V/m
vmin, vmax = 0.0, float(E_mag_FEM.max())
norm = Normalize(vmin=vmin, vmax=vmax)
mappable = cm.ScalarMappable(norm=norm, cmap=colmap)
mappable.set_array([])  # requerido por Matplotlib
cb = plt.colorbar(mappable, ax=ax, pad=0.06, shrink=0.85)
cb.set_label('|E| [V/m]', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Campo_E_FEM_12.png"), dpi=600)
print("Figura guardada en 'results_malla_2M/Campo_E_FEM_12.png'")
plt.show()

# Graficamos el campo eléctrico analítico

colmap = plt.get_cmap("jet")
idx= np.where(mask)[0]
i2m.plotmesh(np.column_stack([coords, E_mag_analytic /np.max(E_mag_analytic)]), cells1,'(x<0) & (x>-0.3)',cmap= colmap)
ax = plt.gca()
ax.view_init(elev=0, azim=0)
ax.quiver(coords[idx,0], coords[idx,1], coords[idx,2], E_matrix_analytic[idx,0], E_matrix_analytic[idx,1], E_matrix_analytic[idx,2], 
          length=0.4, normalize=True, cmap=colmap, color='w', linewidth=0.2)
ax.set_xlabel('')                          # sin nombre
ax.set_xticks([])                          # sin ticks
ax.tick_params(axis='x', which='both',     # sin marcas ni etiquetas
               labelbottom=False, length=0)

# === Título y ejes con unidades ===
ax.set_title('Campo eléctrico analítico', fontsize=14, pad=12)
ax.set_ylabel('y [cm]', fontsize=12, labelpad=8)
ax.set_zlabel('z [cm]', fontsize=12, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=10)

# Colorbar con etiqueta en V/m
vmin, vmax = 0.0, float(E_mag_analytic.max())
norm = Normalize(vmin=vmin, vmax=vmax)
mappable = cm.ScalarMappable(norm=norm, cmap=colmap)
mappable.set_array([])  # requerido por Matplotlib
cb = plt.colorbar(mappable, ax=ax, pad=0.06, shrink=0.85)
cb.set_label('|E| [V/m]', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Campo_E_analitico_12.png"), dpi=600)
print("Figura guardada en 'results_malla_2M/Campo_E_analitico_12.png'")
plt.show()



