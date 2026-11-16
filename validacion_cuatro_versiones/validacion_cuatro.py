
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# Rutas de la Practica 02
BASE = Path(__file__).resolve().parent
PROYECTO = BASE.parent
CODIGO = PROYECTO / "codigo"
SALIDAS = BASE / "salidas"
ARCHIVO_CSV = BASE / "resultados_validacion_cuatro.csv"

sys.path.insert(0, str(CODIGO))

from filtros_python import crear_mascara, convolucion_python
from filtros_pycuda import FiltroPyCUDA

TOLERANCIA = 1e-4
HILOS = 256

FILTROS = [
    ("Media_3x3", "media", 3),
    ("Media_9x9", "media", 9),
    ("Media_21x21", "media", 21),
    ("Sharpen_3x3", "sharpen", 3),
    ("Laplaciano_3x3", "laplaciano", 3),
]


def cargar_binario(ruta, forma):
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe el archivo: {ruta}"
        )

    datos = np.fromfile(ruta, dtype=np.float32)

    if datos.size != np.prod(forma):
        raise ValueError(
            f"Tamano incorrecto en {ruta.name}: "
            f"{datos.size} valores"
        )

    return datos.reshape(forma)


def comparar(a, b):
    if a.shape != b.shape:
        raise ValueError("Las dimensiones son diferentes")

    if not np.isfinite(a).all():
        raise ValueError("La primera salida contiene NaN o infinito")

    if not np.isfinite(b).all():
        raise ValueError("La segunda salida contiene NaN o infinito")

    # Calcular diferencias en float64 para evitar
    # redondeos adicionales durante la comparacion.
    diferencias = np.abs(
        a.astype(np.float64) - b.astype(np.float64)
    )

    error_maximo = float(np.max(diferencias))
    error_promedio = float(np.mean(diferencias))

    return error_maximo, error_promedio


def ejecutar():
    if not SALIDAS.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta: {SALIDAS}"
        )

    casos = [
        (
            "matriz_5x5",
            np.arange(1, 26, dtype=np.float32).reshape(5, 5),
        ),
        (
            "irregular_53x37",
            (np.arange(53 * 37) % 251)
            .astype(np.float32)
            .reshape(37, 53),
        ),
    ]

    motor = FiltroPyCUDA()
    registros = []
    total_correctas = 0
    total_incorrectas = 0

    print("=" * 75)
    print("VALIDACION NUMERICA DE CUATRO IMPLEMENTACIONES")
    print("C/C++ | CUDA C | Python | PyCUDA")
    print(f"Tolerancia absoluta: {TOLERANCIA}")
    print("=" * 75)

    for nombre_caso, entrada in casos:
        print(f"\nCASO: {nombre_caso}")
        print(f"Dimensiones: {entrada.shape}")

        for nombre_filtro, tipo, tamano in FILTROS:
            mascara = crear_mascara(tipo, tamano)

            base = f"{nombre_caso}_{nombre_filtro}"

            salida_cpp = cargar_binario(
                SALIDAS / f"{base}_cpp.bin",
                entrada.shape,
            )

            salida_cuda = cargar_binario(
                SALIDAS / f"{base}_cuda.bin",
                entrada.shape,
            )

            salida_python = convolucion_python(
                entrada, mascara
            )

            salida_pycuda, _ = motor.ejecutar(
                entrada,
                mascara,
                hilos=HILOS,
                repeticiones=1,
            )

            versiones = {
                "C/C++": salida_cpp,
                "CUDA C": salida_cuda,
                "Python": salida_python,
                "PyCUDA": salida_pycuda,
            }

            print(f"\nFiltro: {nombre_filtro}")

            # Comparar todos los pares:
            # 4 versiones = 6 comparaciones por filtro.
            for (nombre_a, datos_a), (nombre_b, datos_b) in (
                combinations(versiones.items(), 2)
            ):
                error_maximo, error_promedio = comparar(
                    datos_a, datos_b
                )

                correcto = error_maximo <= TOLERANCIA
                estado = "CORRECTO" if correcto else "INCORRECTO"

                if correcto:
                    total_correctas += 1
                else:
                    total_incorrectas += 1

                registros.append({
                    "caso": nombre_caso,
                    "alto": entrada.shape[0],
                    "ancho": entrada.shape[1],
                    "filtro": nombre_filtro,
                    "version_a": nombre_a,
                    "version_b": nombre_b,
                    "hilos_gpu": HILOS,
                    "error_maximo": f"{error_maximo:.10f}",
                    "error_promedio": f"{error_promedio:.10f}",
                    "tolerancia": TOLERANCIA,
                    "estado": estado,
                })

                print(
                    f"  {nombre_a:7s} vs {nombre_b:7s} | "
                    f"max={error_maximo:.8f} | "
                    f"prom={error_promedio:.8f} | "
                    f"{estado}"
                )

    campos = [
        "caso",
        "alto",
        "ancho",
        "filtro",
        "version_a",
        "version_b",
        "hilos_gpu",
        "error_maximo",
        "error_promedio",
        "tolerancia",
        "estado",
    ]

    with ARCHIVO_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=campos
        )
        escritor.writeheader()
        escritor.writerows(registros)

    print("\n" + "=" * 75)
    print("RESUMEN FINAL")
    print(f"Comparaciones correctas:   {total_correctas}")
    print(f"Comparaciones incorrectas: {total_incorrectas}")
    print(f"Total de comparaciones:    {len(registros)}")
    print(f"Archivo CSV: {ARCHIVO_CSV}")

    if total_incorrectas == 0:
        print("VALIDACION DE LAS CUATRO VERSIONES: CORRECTA")
    else:
        print("VALIDACION DE LAS CUATRO VERSIONES: FALLIDA")
        sys.exit(1)


if __name__ == "__main__":
    ejecutar()
