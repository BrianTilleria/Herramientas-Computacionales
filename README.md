
 Estimulación eléctrica intracraneal con electrodos profundos
=====================================


En este proyecto se simulan estimulaciones eléctricas en un modelo de cabeza de 3 capas (cerebro, líquido cefalorraquídeo y cráneo).
Se calcula la solución analítica, dado que la geometría esférica lo permite, y la solución con el método de elementos finitos.
Para poder hacer una comparación justa, se generan mallados con el programa de software libre GMESH y se calculas los potenciales eléctricos
por FEM y analítico en los mismos nodos. Los campos eléctricos son generados en FEniCS para ambos casos.

La técnica de estimulación es por interferencia temporal. Básicamente es generar un batido entre dos sinusoides con una pequeña diferencia 
entre las frecuencias. Esta baja frecuencia puede modular la actividad neuronal y llegado al caso puede desatar crisis epilépticas si nos 
encontramos cerca de la zona epileptógena superando cierto umbral.
Los códigos implementan 5 estimulaciones y se calculan errores entre los cálculos analíticos y FEM en una determinada región de interés (ROI).



 Instalación del entorno virtual
=====================================

Para reproducir este proyecto se recomienda usar Conda. 
Se provee un archivo `environment.yml` que contiene todas las 
dependencias necesarias (con versiones específicas).

1. Instalar Miniconda o Anaconda (si aún no lo tienes).
   https://docs.conda.io/en/latest/miniconda.html

2. Clonar este proyecto y ubicarse en la carpeta principal.

3. Crear el entorno virtual a partir del archivo `environment.yml`:
   conda env create -f environment.yml

4. Activar el entorno:
   conda activate fenics_env

5. Verificar la instalación ejecutando:
   python --version
   conda list
   
   

Instrucciones de uso:
================================
1. En la carpeta donde se encuentran las funciones se debe descargar los archivos correspondiente a las mallas.
Los archivos son grandes y no entran en el repositorio, se compartirán mediante drive. Los archivos son:

  -malla_2M.xml

  -malla_2M_facet_region.xml

  -malla_2M_physical_region.xml

2. Ejecutar los códigos:

  -run_compute_E_analytic.py

  -run_compute_fem.py

  -run_compute_error.py

  -run_plots.py

3. Los resultados son almacenados en una carpeta con el nombre de la malla. Se guardan las figuras que se presentan en el informe.
Además, se guardan los potenciales y campos eléctricos de las soluciones FEM y analítica.


Autor: Brian Tilleria
