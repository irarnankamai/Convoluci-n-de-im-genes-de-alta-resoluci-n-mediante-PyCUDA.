
# PRÁCTICA 02 — CONVOLUCIÓN DE IMÁGENES DE ALTA RESOLUCIÓN MEDIANTE PYCUDA

**UNIVERSIDAD POLITÉCNICA SALESIANA (UPS)**

**Asignatura:** Computación Paralela  
**Integrantes:** Irar Nankamai y Noe Ayavaca  
**Tecnologías:** Python, NumPy, CUDA C y PyCUDA  
**Entorno:** Windows, Visual Studio Code y GPU NVIDIA

---

## 1. Descripción del proyecto

Esta práctica desarrolla e implementa filtros de convolución sobre imágenes de alta resolución en escala de grises utilizando dos enfoques:

- **Python secuencial:** implementación mediante bucles explícitos que recorren los píxeles y los coeficientes de cada máscara.
- **PyCUDA:** implementación paralela mediante un kernel propio escrito en CUDA C y ejecutado desde Python con `pycuda.compiler.SourceModule`.

El proyecto permite analizar el rendimiento del procesamiento de imágenes mediante GPU, comprobar la precisión numérica y comparar los resultados con los registros históricos de C/C++ secuencial y CUDA C obtenidos en la Práctica 01.

### Objetivos

1. Implementar filtros de convolución en Python secuencial y PyCUDA.
2. Validar los resultados numéricos utilizando matrices pequeñas e imágenes de dimensiones irregulares.
3. Evaluar el rendimiento con imágenes de 2000 × 2000, 4000 × 4000 y 6000 × 6000 píxeles.
4. Analizar el efecto del tamaño de la máscara y la cantidad de hilos por bloque.
5. Comparar los resultados experimentales con los registros de la Práctica 01.
6. Documentar el procedimiento para facilitar su reproducción.

**Alcance:** la validación directa de la Práctica 02 corresponde a Python y PyCUDA. La comparación numérica conjunta de las cuatro implementaciones todavía no se ha ejecutado sobre una misma entrada con las cuatro salidas disponibles.

---

## 2. Requisitos del sistema

### 2.1. Hardware y software

| Componente | Configuración utilizada |
|---|---|
| Sistema operativo | Windows |
| Editor | Visual Studio Code |
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| Pillow | 13.0.0 |
| Matplotlib | 3.11.2 |
| PyCUDA | 2026.1 |
| GPU | NVIDIA GeForce RTX 3050 |
| Memoria GPU | 4096 MiB |
| Controlador NVIDIA | 621.01 |
| CUDA Toolkit | 13.5 |
| NVCC | 13.5.37 |
| Compilador anfitrión | Microsoft Visual C++ (`cl.exe`) |

**Importante:** para ejecutar PyCUDA se necesita una GPU NVIDIA compatible, sus controladores, CUDA Toolkit y un entorno de compilación compatible.

Los datos exactos de CPU, RAM y versión de Windows no quedaron registrados en las mediciones disponibles.

### 2.2. Verificar CUDA y la GPU

Abrir **Developer PowerShell for Visual Studio** y ejecutar:

```powershell
nvidia-smi
nvcc --version
where.exe cl
```

Estos comandos permiten comprobar que la GPU, el compilador CUDA y el compilador de Microsoft están disponibles.

---

## 3. Estructura del proyecto

```text
PRACTICA2/
│
├── codigo/
│   ├── preparar_imagenes.py
│   ├── filtros_python.py
│   ├── filtros_pycuda.py
│   ├── validaciones.py
│   ├── experimentos.py
│   ├── generar_imagenes.py
│   └── comparacion_cpu_gpu.py
│
├── imagenes/
│   ├── imagen_2000.pgm
│   ├── imagen_4000.pgm
│   └── imagen_6000.pgm
│
├── resultados/
│   ├── mediciones_pycuda.csv
│   ├── resumen_pycuda.csv
│   ├── mediciones_python_recortes.csv
│   ├── comparacion_cpu_gpu_detalle.csv
│   ├── comparacion_cpu_gpu_resumen.csv
│   ├── Original_2000.png
│   ├── Media_3x3_PyCUDA.png
│   ├── Media_9x9_PyCUDA.png
│   ├── Media_21x21_PyCUDA.png
│   ├── Sharpen_3x3_PyCUDA.png
│   ├── Laplaciano_3x3_PyCUDA.png
│   └── *_PyCUDA_float32.npy
│
├── graficas/
│   ├── kernel_pycuda.png
│   └── total_pycuda.png
│
├── README.md
├── requirements.txt
└── dependencias_entorno.txt
```

### Descripción de los programas

| Archivo | Función |
|---|---|
| `preparar_imagenes.py` | Preparación de imágenes de entrada |
| `filtros_python.py` | Convolución secuencial en Python |
| `filtros_pycuda.py` | Kernel de convolución y ejecución PyCUDA |
| `validaciones.py` | Comprobación de resultados numéricos |
| `experimentos.py` | Mediciones de rendimiento y gráficas |
| `generar_imagenes.py` | Generación de imágenes filtradas |
| `comparacion_cpu_gpu.py` | Comparación Python y PyCUDA sobre recortes |

Los directorios `.venv/` y `__pycache__/` son generados localmente y no deben publicarse en GitHub.

---

## 4. Instalación y configuración

### Paso 1. Abrir el proyecto

En PowerShell:

```powershell
cd "$HOME\OneDrive\Desktop\PRACTICA2"
```

La ruta corresponde al equipo utilizado durante la práctica. En otro equipo debe sustituirse por la ubicación donde se descargue el proyecto.

### Paso 2. Crear un entorno virtual

```powershell
py -3.11 -m venv .venv
```

### Paso 3. Instalar las dependencias

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Las bibliotecas principales utilizadas son:

- NumPy.
- Pillow.
- Matplotlib.
- PyCUDA.

**Nota:** antes de publicar, verificar que `requirements.txt` contenga las dependencias realmente utilizadas y que no incluya paquetes innecesarios.

El archivo `dependencias_entorno.txt` conserva un registro del entorno Python utilizado durante el desarrollo.

### Paso 4. Verificar Python

```powershell
.\.venv\Scripts\python.exe --version
```

### Paso 5. Verificar PyCUDA

```powershell
.\.venv\Scripts\python.exe -c "import pycuda.driver as cuda; cuda.init(); d=cuda.Device(0); print('GPU:', d.name()); print('Compute Capability:', d.compute_capability())"
```

Si el comando muestra el nombre de la GPU y su capacidad de cómputo, PyCUDA puede detectar el dispositivo.

---

## 5. Compilación del código CUDA

La Práctica 02 utiliza un kernel escrito en CUDA C dentro del archivo:

`codigo/filtros_pycuda.py`

El kernel se compila mediante:

```python
from pycuda.compiler import SourceModule
```

**No es necesario ejecutar manualmente `nvcc` para compilar el kernel de esta práctica.**

PyCUDA utiliza la infraestructura de compilación CUDA al ejecutar el programa.

Para ello deben estar instalados:

1. CUDA Toolkit.
2. Controlador NVIDIA compatible.
3. Python y PyCUDA.
4. Herramientas de compilación de Microsoft Visual Studio compatibles con CUDA.

En Windows se recomienda abrir **Developer PowerShell for Visual Studio** antes de ejecutar los programas.

### Verificación práctica de la compilación

Ejecutar:

```powershell
.\.venv\Scripts\python.exe codigo\validaciones.py
```

Este comando inicializa PyCUDA, compila el kernel y ejecuta las pruebas de validación.

Si termina con:

```text
VALIDACIONES COMPLETADAS
```

la ejecución y las validaciones habrán finalizado correctamente.

---

## 6. Imágenes utilizadas

Se emplean tres imágenes PGM en escala de grises:

| Imagen | Resolución | Píxeles |
|---|---:|---:|
| `imagen_2000.pgm` | 2000 × 2000 | 4 000 000 |
| `imagen_4000.pgm` | 4000 × 4000 | 16 000 000 |
| `imagen_6000.pgm` | 6000 × 6000 | 36 000 000 |

Las imágenes de entrada utilizadas en ambas prácticas se verificaron como idénticas a nivel de archivo.

Para reproducir los resultados deben utilizarse las mismas imágenes y máscaras.

---

## 7. Filtros implementados

### 7.1. Filtro Media

Se implementaron máscaras de:

- 3 × 3.
- 9 × 9.
- 21 × 21.

Cada coeficiente tiene el valor:

```text
1 / (k × k)
```

El filtro Media suaviza la imagen mediante el promedio de los valores vecinos.

### 7.2. Filtro Sharpen 3 × 3

```text
 0  -1   0
-1   5  -1
 0  -1   0
```

Permite resaltar los detalles y mejorar la nitidez.

### 7.3. Filtro Laplaciano 3 × 3

```text
 0   1   0
 1  -4   1
 0   1   0
```

Permite resaltar cambios de intensidad y bordes.

### 7.4. Convenciones numéricas

- Entrada de cálculo y salida en `float32`.
- Extensión con ceros fuera de los límites de la imagen (*zero padding*).
- Una posición de salida por hilo CUDA.
- Acceso a memoria global de GPU.
- Salidas del mismo tamaño que las entradas.
- Validación numérica antes de convertir a PNG.
- Compilación del kernel PyCUDA con `--fmad=false` para reducir diferencias de operaciones fusionadas.

---

## 8. Implementación secuencial en Python

**Archivo:** `codigo/filtros_python.py`

La convolución se realiza mediante bucles explícitos sobre las filas, columnas y elementos de la máscara.

No se utilizan OpenCV, SciPy, Numba ni multiprocessing para ejecutar la convolución.

Debido al coste de procesamiento de los bucles interpretados, las mediciones de Python secuencial se realizaron sobre recortes reales de **64 × 64 píxeles**.

No se registraron tiempos completos de Python secuencial para las imágenes de 2000², 4000² y 6000².

---

## 9. Implementación paralela con PyCUDA

**Archivo:** `codigo/filtros_pycuda.py`

El kernel CUDA calcula un píxel de salida por hilo y comprueba los límites de la imagen.

Se utilizaron configuraciones de:

- 128 hilos por bloque.
- 256 hilos por bloque.
- 512 hilos por bloque.

### Etapas de ejecución

1. **H2D:** transferencia de los datos desde la CPU a la GPU.
2. **Kernel:** ejecución de la convolución.
3. **D2H:** transferencia del resultado desde la GPU a la CPU.
4. **Total GPU:** tiempo medido de transferencias y procesamiento.

La compilación inicial, las asignaciones de memoria y el calentamiento no forman parte de las cinco repeticiones medidas.

---

## 10. Validación de resultados

### Ejecutar las validaciones

```powershell
.\.venv\Scripts\python.exe codigo\validaciones.py
```

### Ejecutar sin GPU

```powershell
.\.venv\Scripts\python.exe codigo\validaciones.py --sin-gpu
```

### Resultados obtenidos

| Prueba | Resultado |
|---|---|
| Media 3 × 3 manual frente a Python | CORRECTO |
| Media 3 × 3 Python/PyCUDA, 5 × 5 | CORRECTO |
| Media 9 × 9 Python/PyCUDA, 5 × 5 | CORRECTO |
| Media 21 × 21 Python/PyCUDA, 5 × 5 | CORRECTO |
| Sharpen 3 × 3 Python/PyCUDA, 5 × 5 | CORRECTO |
| Laplaciano 3 × 3 Python/PyCUDA, 5 × 5 | CORRECTO |
| Imagen irregular 53 × 37, Media | CORRECTO |
| Imagen irregular 53 × 37, Sharpen | CORRECTO |
| Imagen irregular 53 × 37, Laplaciano | CORRECTO |
| Pruebas con 128, 256 y 512 hilos | CORRECTO |

### Errores numéricos

**Media 3 × 3 manual frente a Python:**

```text
Error máximo: 0.00000191
Error medio:  0.00000015
```

**Python frente a PyCUDA:**

```text
Error máximo: 0.00000000
Error medio:  0.00000000
```

Los errores de Python/PyCUDA corresponden a las comparaciones ejecutadas y mostradas por el programa.

**Tolerancia absoluta:** `1e-4`.

### Resultado final

```text
VALIDACIONES COMPLETADAS
```

**Limitación:** estas pruebas demuestran concordancia entre Python y PyCUDA. La validación numérica simultánea de C/C++ secuencial, CUDA C, Python y PyCUDA todavía está pendiente.

---

## 11. Experimentos de alta resolución

### Ejecutar los experimentos GPU

```powershell
.\.venv\Scripts\python.exe codigo\experimentos.py --modo gpu --repeticiones 5
```

Se realizaron nueve configuraciones con cinco repeticiones cada una.

**Total: 45 mediciones GPU.**

### Configuraciones utilizadas

| Prueba | Filtro | Resolución | Hilos |
|---|---|---:|---:|
| P1 | Media 3 × 3 | 4000 × 4000 | 256 |
| P2 | Sharpen 3 × 3 | 4000 × 4000 | 256 |
| P3 | Laplaciano 3 × 3 | 4000 × 4000 | 256 |
| P4 | Media 9 × 9 | 2000 × 2000 | 256 |
| P5 | Media 9 × 9 | 4000 × 4000 | 256 |
| P6 | Media 9 × 9 | 6000 × 6000 | 256 |
| P7 | Media 21 × 21 | 4000 × 4000 | 256 |
| P8 | Media 9 × 9 | 4000 × 4000 | 128 |
| P9 | Media 9 × 9 | 4000 × 4000 | 512 |

### Resultados de PyCUDA

Tiempos en milisegundos. Las columnas muestran las medianas de cinco repeticiones y la desviación estándar del tiempo total.

| Prueba | H2D | Kernel | D2H | Total GPU | Desv. total |
|---|---:|---:|---:|---:|---:|
| P1 | 11.996 | 1.335 | 10.750 | 23.959 | 3.129 |
| P2 | 11.276 | 1.286 | 10.373 | 23.177 | 3.168 |
| P3 | 12.855 | 1.085 | 12.007 | 25.880 | 3.432 |
| P4 | 4.048 | 1.422 | 3.556 | 9.426 | 0.895 |
| P5 | 13.945 | 5.390 | 12.069 | 31.456 | 3.674 |
| P6 | 27.343 | 12.042 | 22.891 | 63.507 | 3.855 |
| P7 | 13.045 | 28.832 | 13.220 | 55.251 | 1.888 |
| P8 | 13.379 | 5.420 | 11.223 | 30.072 | 2.730 |
| P9 | 11.508 | 5.413 | 10.005 | 26.745 | 2.651 |

**Nota:** la suma de las medianas de H2D, kernel y D2H puede diferir de la mediana del tiempo total, porque se calculan de forma independiente.

### Archivos generados

```text
resultados/mediciones_pycuda.csv
resultados/resumen_pycuda.csv
```

---

## 12. Experimentos de Python secuencial

### Ejecutar las mediciones

```powershell
.\.venv\Scripts\python.exe codigo\experimentos.py --modo cpu --recorte 64 --repeticiones 5
```

Se procesan recortes de 64 × 64 píxeles con cinco filtros y cinco repeticiones.

**Total: 25 mediciones CPU.**

### Comparar Python y PyCUDA

```powershell
.\.venv\Scripts\python.exe codigo\comparacion_cpu_gpu.py
```

### Resultados de la comparación

| Filtro | Python CPU (ms) | PyCUDA kernel (ms) | PyCUDA total (ms) | Aceleración total |
|---|---:|---:|---:|---:|
| Media 3 × 3 | 33.4722 | 0.023616 | 0.0718 | 466.19× |
| Sharpen 3 × 3 | 32.9647 | 0.024576 | 0.0788 | 418.33× |
| Laplaciano 3 × 3 | 33.9645 | 0.023552 | 0.0775 | 438.25× |
| Media 9 × 9 | 238.5171 | 0.024576 | 0.0834 | 2859.92× |
| Media 21 × 21 | 1132.3828 | 0.034816 | 0.1042 | 10867.40× |

En las cinco comparaciones se reportó un error máximo de `0.0`.

La aceleración se calcula mediante:

```text
Aceleración = Tiempo Python CPU / Tiempo total PyCUDA
```

**Interpretación:** las aceleraciones corresponden a Python interpretado frente a PyCUDA sobre recortes de 64 × 64 píxeles. No deben extrapolarse a imágenes grandes ni interpretarse como una comparación contra C++ compilado.

Los tiempos GPU no incluyen la compilación inicial ni la asignación de memoria.

### Archivos generados

```text
resultados/mediciones_python_recortes.csv
resultados/comparacion_cpu_gpu_detalle.csv
resultados/comparacion_cpu_gpu_resumen.csv
```

---

## 13. Generación de imágenes filtradas

Ejecutar:

```powershell
.\.venv\Scripts\python.exe codigo\generar_imagenes.py
```

Se generan las siguientes imágenes:

```text
resultados/Original_2000.png
resultados/Media_3x3_PyCUDA.png
resultados/Media_9x9_PyCUDA.png
resultados/Media_21x21_PyCUDA.png
resultados/Sharpen_3x3_PyCUDA.png
resultados/Laplaciano_3x3_PyCUDA.png
```

También se generan las salidas numéricas:

```text
resultados/Media_3x3_PyCUDA_float32.npy
resultados/Media_9x9_PyCUDA_float32.npy
resultados/Media_21x21_PyCUDA_float32.npy
resultados/Sharpen_3x3_PyCUDA_float32.npy
resultados/Laplaciano_3x3_PyCUDA_float32.npy
```

Los archivos `.npy` contienen valores `float32` antes del recorte de visualización.

**Importante:** los PNG son imágenes para inspección visual. No deben utilizarse como sustituto de las salidas numéricas al calcular errores de precisión.

---

## 14. Generación de gráficas

Ejecutar:

```powershell
.\.venv\Scripts\python.exe codigo\experimentos.py --modo graficas
```

Se generan dos gráficas:

- `graficas/kernel_pycuda.png`: tiempos de ejecución del kernel.
- `graficas/total_pycuda.png`: tiempos totales de procesamiento GPU.

Estas gráficas permiten analizar las diferencias entre las nueve configuraciones experimentales.

---

## 15. Comparación histórica con la Práctica 01

La Práctica 01 implementó filtros de convolución mediante:

- C/C++ secuencial.
- CUDA C.

La Práctica 02 implementó:

- Python secuencial.
- PyCUDA.

Se utilizaron los registros anteriores de la Práctica 01 sin modificar sus programas ni repetir sus experimentos.

### 15.1. Diferencias de hardware

| Característica | Práctica 01 | Práctica 02 |
|---|---|---|
| GPU | NVIDIA RTX 3070 | NVIDIA RTX 3050 |
| Implementación paralela | CUDA C | PyCUDA |
| Implementación secuencial | C/C++ | Python |
| Datos de entrada | Imágenes PGM | Las mismas imágenes PGM |

Debido a que se utilizaron GPU distintas y existen diferencias de implementación y metodología, los resultados se presentan como una **comparación histórica descriptiva**, no como una prueba controlada de superioridad de una tecnología.

### 15.2. Comparación de tiempos GPU

Tiempos en milisegundos.

| Prueba | CUDA C kernel | PyCUDA kernel | CUDA C total | PyCUDA total |
|---|---:|---:|---:|---:|
| P1 | 10.814 | 1.335 | 26.592 | 23.959 |
| P2 | 7.275 | 1.286 | 21.684 | 23.177 |
| P3 | 7.278 | 1.085 | 21.520 | 25.880 |
| P4 | 16.430 | 1.422 | 20.148 | 9.426 |
| P5 | 81.423 | 5.390 | 95.619 | 31.456 |
| P6 | 375.164 | 12.042 | 423.663 | 63.507 |
| P7 | 524.638 | 28.832 | 546.191 | 55.251 |
| P8 | 106.610 | 5.420 | 121.079 | 30.072 |
| P9 | 120.128 | 5.413 | 145.318 | 26.745 |

PyCUDA registró un tiempo total menor en **siete de las nueve configuraciones**. CUDA C registró un tiempo total menor en P2 y P3.

**Estos resultados no demuestran que PyCUDA sea intrínsecamente más rápido que CUDA C**, porque las pruebas se ejecutaron con hardware distinto y las condiciones no fueron idénticas.

### 15.3. Tiempos de C/C++ secuencial

| Prueba | Tiempo CPU C/C++ (ms) |
|---|---:|
| P1 | 230.965 |
| P2 | 288.210 |
| P3 | 274.652 |
| P4 | 329.315 |
| P5 | 1309.218 |
| P6 | 2955.924 |
| P7 | 6645.726 |
| P8 | 1313.482 |
| P9 | 1310.003 |

Estos tiempos proceden de la Práctica 01.

No se comparan directamente con los tiempos Python de 64 × 64 píxeles, porque corresponden a resoluciones diferentes.

### 15.4. Validación conjunta de las cuatro versiones

Las validaciones internas de ambas prácticas se completaron de manera independiente.

Sin embargo, todavía no se dispone de una comparación numérica directa de las cuatro salidas:

1. C/C++ secuencial.
2. CUDA C.
3. Python secuencial.
4. PyCUDA.

Para completar esta comprobación se necesitan las cuatro salidas numéricas de una misma imagen o matriz, antes de convertirlas a PNG, y calcular el error absoluto máximo y medio frente a C/C++ secuencial.

---

## 16. Análisis de rendimiento

### 16.1. Influencia de la resolución

Con el filtro Media 9 × 9 y 256 hilos:

| Resolución | Kernel PyCUDA (ms) | Total GPU (ms) |
|---|---:|---:|
| 2000 × 2000 | 1.422 | 9.426 |
| 4000 × 4000 | 5.390 | 31.456 |
| 6000 × 6000 | 12.042 | 63.507 |

El tiempo de procesamiento aumenta al incrementar el número de píxeles.

### 16.2. Influencia del tamaño de la máscara

Para imágenes de 4000 × 4000 y 256 hilos:

| Máscara | Kernel PyCUDA (ms) |
|---|---:|
| Media 3 × 3 | 1.335 |
| Media 9 × 9 | 5.390 |
| Media 21 × 21 | 28.832 |

El incremento del tamaño de la máscara aumenta la cantidad de operaciones por píxel.

### 16.3. Influencia de los hilos por bloque

Para Media 9 × 9 sobre 4000 × 4000:

| Hilos | Kernel PyCUDA (ms) | Total GPU (ms) |
|---|---:|---:|
| 128 | 5.420 | 30.072 |
| 256 | 5.390 | 31.456 |
| 512 | 5.413 | 26.745 |

Los tiempos del kernel son similares. La variación de los tiempos totales impide establecer una configuración universalmente superior.

### 16.4. Transferencias de memoria

Las transferencias entre CPU y GPU representan una parte importante del tiempo total en varias pruebas.

Por ello, es necesario considerar tanto el tiempo del kernel como los tiempos H2D, D2H y total GPU.

---

## 17. Limitaciones

- Python secuencial se midió sobre recortes de 64 × 64 píxeles.
- No se dispone de tiempos medidos de Python secuencial para las imágenes completas.
- La comparación histórica utiliza GPU distintas: RTX 3070 y RTX 3050.
- Los tiempos GPU no incluyen compilación inicial ni asignación de memoria.
- Las mediciones pueden variar según la carga del equipo y otras condiciones de ejecución.
- Los archivos PNG no sustituyen la comparación numérica de los valores `float32`.
- La validación conjunta de las cuatro implementaciones todavía está pendiente.
- No se registraron todos los datos de CPU, RAM y versión exacta del sistema operativo.

Estas limitaciones se documentan para evitar conclusiones que no estén respaldadas por los experimentos realizados.

---

## 18. Archivos de resultados

| Archivo | Contenido |
|---|---|
| `mediciones_pycuda.csv` | 45 mediciones GPU |
| `resumen_pycuda.csv` | Nueve configuraciones y estadísticas |
| `mediciones_python_recortes.csv` | 25 mediciones Python |
| `comparacion_cpu_gpu_detalle.csv` | Comparaciones individuales Python/PyCUDA |
| `comparacion_cpu_gpu_resumen.csv` | Aceleraciones y errores |
| `*_PyCUDA.png` | Imágenes filtradas |
| `*_PyCUDA_float32.npy` | Resultados numéricos sin recorte |
| `kernel_pycuda.png` | Gráfica de tiempos del kernel |
| `total_pycuda.png` | Gráfica de tiempos totales |

---

## 19. Secuencia para reproducir la práctica

Después de instalar y configurar el entorno, ejecutar los siguientes comandos en este orden:

### 1. Validar los filtros

```powershell
.\.venv\Scripts\python.exe codigo\validaciones.py
```

### 2. Ejecutar los experimentos GPU

```powershell
.\.venv\Scripts\python.exe codigo\experimentos.py --modo gpu --repeticiones 5
```

### 3. Ejecutar los experimentos Python

```powershell
.\.venv\Scripts\python.exe codigo\experimentos.py --modo cpu --recorte 64 --repeticiones 5
```

### 4. Comparar Python y PyCUDA

```powershell
.\.venv\Scripts\python.exe codigo\comparacion_cpu_gpu.py
```

### 5. Generar las imágenes

```powershell
.\.venv\Scripts\python.exe codigo\generar_imagenes.py
```

### 6. Generar las gráficas

```powershell
.\.venv\Scripts\python.exe codigo\experimentos.py --modo graficas
```

**Nota:** ejecutar nuevamente los experimentos puede sobrescribir archivos de resultados existentes y producir tiempos diferentes debido a las condiciones del equipo.

---

## 20. Conclusiones

Se implementaron filtros de convolución mediante Python secuencial y PyCUDA, utilizando máscaras Media, Sharpen y Laplaciano.

Las validaciones realizadas confirmaron la concordancia numérica entre Python y PyCUDA en matrices pequeñas y dimensiones irregulares, con una tolerancia absoluta de `1e-4`.

Los experimentos demostraron que el rendimiento depende de la resolución de las imágenes, el tamaño de las máscaras, la configuración de hilos y las transferencias de memoria.

En las pruebas de 64 × 64 píxeles, PyCUDA registró tiempos menores que Python secuencial. Sin embargo, estos resultados no representan una comparación contra C++ optimizado.

La comparación histórica con la Práctica 01 permitió analizar los resultados de CUDA C y PyCUDA, considerando las diferencias de hardware y metodología. Para demostrar la equivalencia numérica de las cuatro implementaciones todavía se requiere una validación conjunta.

---

## 21. Autores

**Universidad Politécnica Salesiana (UPS)**  
**Asignatura:** Computación Paralela

**Integrantes:**
- Irar Nankamai
- Noe Ayavaca

**Proyecto:** Práctica 02 — Convolución de imágenes de alta resolución mediante PyCUDA.

---

**UNIVERSIDAD POLITÉCNICA SALESIANA — COMPUTACIÓN PARALELA**

