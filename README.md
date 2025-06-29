# Asignación de Escritorios Optimizada con Algoritmo Genético
## Características Principales
* **Algoritmo Genético:** Utiliza un algoritmo genético robusto para explorar soluciones y encontrar asignaciones que minimicen penalizaciones.
* **Gestión de Restricciones:** Maneja eficazmente restricciones duras (ej. capacidad de escritorios, asignaciones permitidas) y suaves (ej. días preferidos, minimización de aislamiento grupal).
* **Reportes Exhaustivos:** Genera reportes detallados en formato **JSON**, **Excel** y **PDF**, incluyendo un resumen de **KPIs clave** (Key Performance Indicators) para evaluar la calidad de la solución.
* **Visualizaciones:** Incluye gráficos claros para la capacidad de escritorios utilizada por día y el rendimiento de KPIs importantes.
* **Carga de Datos Flexible:** La configuración de empleados, escritorios, días, grupos y zonas se carga fácilmente desde un archivo JSON.

## Cómo Ejecutar el Proyecto en Google Colab

Este proyecto está diseñado para ejecutarse cómodamente en Google Colab.

1.  **Abrir en Colab:**
    * Ve a [colab.research.google.com](https://colab.research.google.com/).
    * Selecciona **"File" (Archivo) > "Open notebook" (Abrir cuaderno)**.
    * Ve a la pestaña **"GitHub"** y busca tu repositorio (`tu_usuario/Challenge_ASOCIO25_Los-IOs)`).
    * Navega a `src/main.py` y haz clic para abrirlo.

2.  **Instalar Dependencias:** En las primeras celda del cuaderno de Colab, ejecuta los comandos para instalar las librerías necesarias(ejecutar individualmente cada una de las librerías):
pip install openpyxl xlsxwriter matplotlib seaborn reportlab
pip install pulp
pip install ortools
  
4.  **Ejecutar el Algoritmo:** Ejecuta la celda que contiene el código del algoritmo genético.
   El script te pedirá que **subas tu archivo `data.json`** desde tu computadora (puedes usar los archivo de ejemplo que se encuentra en la carpeta `data/` de este repositorio).

6.  **Ver Resultados:**
    * En la salida de la celda de Colab, verás un **resumen detallado de los KPIs finales** de la asignación óptima.
    * Los reportes completos (JSON, Excel, PDF) y los gráficos (PNG) se generarán y estarán disponibles para descargar en el panel de archivos de Colab.
    * Se verifica que los archivos generados por el código sean los ejemplificados en los archivos que se encuentran en la carpeta `reports/` de este repositorio

## Contribuciones
¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar el algoritmo, añadir nuevas restricciones o mejorar los reportes, no dudes en abrir un `issue` o enviar un `pull request`.

## Licencia
Este proyecto es de código abierto.
