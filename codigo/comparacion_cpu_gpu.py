
"""Comparación justa Python secuencial vs PyCUDA en igual resolución."""

import csv
import statistics
import time
from pathlib import Path

import numpy as np

from filtros_python import crear_mascara, convolucion_python
from filtros_pycuda import FiltroPyCUDA
from preparar_imagenes import cargar_pgm

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

REPETICIONES = 5
RECORTE = 64

PRUEBAS = [
    ("Media", 3),
    ("Sharpen", 3),
    ("Laplaciano", 3),
    ("Media", 9),
    ("Media", 21),
]


def main():
    imagen = cargar_pgm(
        RAIZ / "imagenes" / "imagen_2000.pgm"
    ).astype(np.float32)

    entrada = np.ascontiguousarray(
        imagen[:RECORTE, :RECORTE]
    )

    motor = FiltroPyCUDA()

    registros = []
    resumen = []

    print("COMPARACION PYTHON SECUENCIAL VS PYCUDA")
    print(f"Resolucion: {RECORTE}x{RECORTE}")
    print(f"Repeticiones: {REPETICIONES}")
    print(f"Compilacion excluida: {motor.compilacion_ms:.3f} ms")

    for filtro, tam in PRUEBAS:
        mascara = crear_mascara(filtro, tam)

        # Calentamiento de Python
        convolucion_python(entrada, mascara)

        # PyCUDA incluye calentamiento interno
        salida_gpu, tiempos_gpu = motor.ejecutar(
            entrada,
            mascara,
            hilos=256,
            repeticiones=REPETICIONES
        )

        tiempos_cpu = []
        errores = []

        for repeticion in range(REPETICIONES):
            inicio = time.perf_counter()
            salida_cpu = convolucion_python(entrada, mascara)
            cpu_ms = (time.perf_counter() - inicio) * 1000.0

            error = np.abs(
                salida_cpu.astype(np.float64)
                - salida_gpu.astype(np.float64)
            )

            error_max = float(error.max())
            error_medio = float(error.mean())

            if error_max > 1e-4:
                raise AssertionError(
                    f"Error excesivo: {filtro} {tam}x{tam}"
                )

            gpu = tiempos_gpu[repeticion]

            tiempos_cpu.append(cpu_ms)
            errores.append(error_max)

            registros.append({
                "filtro": filtro,
                "mascara": tam,
                "resolucion": RECORTE,
                "repeticion": repeticion + 1,
                "cpu_ms": cpu_ms,
                "h2d_ms": gpu["h2d_ms"],
                "kernel_ms": gpu["kernel_ms"],
                "d2h_ms": gpu["d2h_ms"],
                "gpu_total_ms": gpu["gpu_total_ms"],
                "error_max": error_max,
                "error_medio": error_medio
            })

        med_cpu = statistics.median(tiempos_cpu)
        med_kernel = statistics.median(
            t["kernel_ms"] for t in tiempos_gpu
        )
        med_total = statistics.median(
            t["gpu_total_ms"] for t in tiempos_gpu
        )

        resumen.append({
            "filtro": filtro,
            "mascara": tam,
            "resolucion": RECORTE,
            "cpu_mediana_ms": med_cpu,
            "kernel_mediana_ms": med_kernel,
            "gpu_total_mediana_ms": med_total,
            "speedup_kernel": med_cpu / med_kernel,
            "speedup_total": med_cpu / med_total,
            "error_max": max(errores)
        })

        print(
            f"{filtro} {tam}x{tam}: "
            f"CPU={med_cpu:.3f} ms | "
            f"Kernel={med_kernel:.3f} ms | "
            f"GPU total={med_total:.3f} ms | "
            f"Speedup total={med_cpu / med_total:.2f}x | "
            f"Error={max(errores):.8f}"
        )

    for nombre, datos in [
        ("comparacion_cpu_gpu_detalle.csv", registros),
        ("comparacion_cpu_gpu_resumen.csv", resumen)
    ]:
        ruta = RESULTADOS / nombre
        with open(ruta, "w", newline="", encoding="utf-8") as archivo:
            writer = csv.DictWriter(
                archivo,
                fieldnames=list(datos[0].keys())
            )
            writer.writeheader()
            writer.writerows(datos)

        print(f"Guardado: {ruta}")

    print("\nCOMPARACION COMPLETADA")


if __name__ == "__main__":
    main()
