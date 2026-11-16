"""Experimentos reproducibles: PyCUDA completo y CPU Python controlado."""
import argparse
import csv
import gc
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from filtros_python import crear_mascara, convolucion_python
from filtros_pycuda import FiltroPyCUDA
from preparar_imagenes import cargar_pgm

RAIZ = Path(__file__).resolve().parent.parent
IMAGENES = RAIZ / 'imagenes'
RESULTADOS = RAIZ / 'resultados'
GRAFICAS = RAIZ / 'graficas'

PRUEBAS = [
    (1, 4000, 'Media', 3, 256),
    (2, 4000, 'Sharpen', 3, 256),
    (3, 4000, 'Laplaciano', 3, 256),
    (4, 2000, 'Media', 9, 256),
    (5, 4000, 'Media', 9, 256),
    (6, 6000, 'Media', 9, 256),
    (7, 4000, 'Media', 21, 256),
    (8, 4000, 'Media', 9, 128),
    (9, 4000, 'Media', 9, 512),
]


def guardar_csv(ruta, registros):
    if not registros:
        return
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(registros[0]))
        writer.writeheader()
        writer.writerows(registros)


def experimentos_gpu(repeticiones):
    motor = FiltroPyCUDA()
    print(f'Compilación inicial PyCUDA: {motor.compilacion_ms:.2f} ms (excluida de benchmarks)')
    registros, resumen = [], []
    for numero, resolucion, filtro, tam, hilos in PRUEBAS:
        ruta = IMAGENES / f'imagen_{resolucion}.pgm'
        if not ruta.exists():
            raise FileNotFoundError(f'Falta {ruta}. Copia los PGM de la Práctica 01.')
        entrada = cargar_pgm(ruta).astype(np.float32)
        if entrada.shape != (resolucion, resolucion):
            raise ValueError(f'Dimensiones inesperadas: {ruta}: {entrada.shape}')
        mascara = crear_mascara(filtro, tam)
        print(f'Prueba {numero}/9: {filtro} {tam}x{tam}, {resolucion}x{resolucion}, {hilos} hilos', flush=True)
        salida, tiempos = motor.ejecutar(entrada, mascara, hilos=hilos, repeticiones=repeticiones)
        if not np.isfinite(salida).all():
            raise ValueError(f'Valores no finitos en prueba {numero}')
        for fila in tiempos:
            registros.append(dict(version='PyCUDA', prueba=numero, resolucion=resolucion,
                                  filtro=filtro, mascara=tam, hilos=hilos,
                                  estado='COMPLETADA', **fila))
        resumen.append(dict(version='PyCUDA', prueba=numero, resolucion=resolucion,
                            filtro=filtro, mascara=tam, hilos=hilos,
                            repeticiones=repeticiones,
                            h2d_mediana_ms=statistics.median(t['h2d_ms'] for t in tiempos),
                            kernel_mediana_ms=statistics.median(t['kernel_ms'] for t in tiempos),
                            d2h_mediana_ms=statistics.median(t['d2h_ms'] for t in tiempos),
                            gpu_total_mediana_ms=statistics.median(t['gpu_total_ms'] for t in tiempos),
                            gpu_total_desv_ms=statistics.stdev(t['gpu_total_ms'] for t in tiempos) if repeticiones > 1 else 0.0))
        guardar_csv(RESULTADOS / 'mediciones_pycuda.csv', registros)
        guardar_csv(RESULTADOS / 'resumen_pycuda.csv', resumen)
        print(f'  Kernel: {resumen[-1]["kernel_mediana_ms"]:.3f} ms | Total: {resumen[-1]["gpu_total_mediana_ms"]:.3f} ms')
        del entrada, salida
        gc.collect()
    return resumen


def experimentos_cpu(recorte, repeticiones):
    """Mide recortes REALES; nunca los presenta como tiempos de imagen completa."""
    imagen = cargar_pgm(IMAGENES / 'imagen_2000.pgm').astype(np.float32)
    if recorte > min(imagen.shape):
        raise ValueError('Recorte demasiado grande')
    entrada = np.ascontiguousarray(imagen[:recorte, :recorte])
    motor = FiltroPyCUDA()
    registros = []
    for filtro, tam in [('Media', 3), ('Sharpen', 3), ('Laplaciano', 3),
                        ('Media', 9), ('Media', 21)]:
        mascara = crear_mascara(filtro, tam)
        gpu, _ = motor.ejecutar(entrada, mascara, hilos=256)
        for rep in range(1, repeticiones + 1):
            t0 = time.perf_counter()
            cpu = convolucion_python(entrada, mascara)
            cpu_ms = (time.perf_counter() - t0) * 1000.0
            error = np.abs(cpu.astype(np.float64) - gpu.astype(np.float64))
            max_error = float(error.max())
            fila = dict(version='Python secuencial', filtro=filtro, mascara=tam,
                        resolucion=recorte, origen='recorte superior izquierdo imagen_2000.pgm',
                        repeticion=rep, cpu_ms=cpu_ms, error_max=max_error,
                        error_medio=float(error.mean()),
                        estado='CORRECTA' if max_error <= 1e-4 else 'ERROR')
            registros.append(fila)
            print(f'{filtro} {tam}x{tam}, {recorte}x{recorte}, rep {rep}: {cpu_ms:.3f} ms, error {max_error:.7f}', flush=True)
            guardar_csv(RESULTADOS / 'mediciones_python_recortes.csv', registros)
            if max_error > 1e-4:
                raise AssertionError('Error CPU/GPU supera tolerancia')
    return registros


def graficar():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ruta = RESULTADOS / 'resumen_pycuda.csv'
    if not ruta.exists():
        print('Primero ejecuta --modo gpu')
        return
    with open(ruta, encoding='utf-8', newline='') as f:
        filas = list(csv.DictReader(f))
    etiquetas = [f'P{r["prueba"]}' for r in filas]
    for clave, nombre, titulo in [
        ('kernel_mediana_ms', 'kernel_pycuda.png', 'PyCUDA: mediana del kernel'),
        ('gpu_total_mediana_ms', 'total_pycuda.png', 'PyCUDA: mediana del tiempo total')
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(etiquetas, [float(r[clave]) for r in filas])
        ax.set_xlabel('Configuración experimental')
        ax.set_ylabel('Tiempo (ms)')
        ax.set_title(titulo)
        ax.grid(axis='y', alpha=.3)
        fig.tight_layout()
        fig.savefig(GRAFICAS / nombre, dpi=180)
        plt.close(fig)
        print('Gráfica:', GRAFICAS / nombre)


def main():
    p = argparse.ArgumentParser(description='Práctica 02: benchmarks PyCUDA y Python secuencial')
    p.add_argument('--modo', choices=['gpu', 'cpu', 'graficas'], required=True)
    p.add_argument('--repeticiones', type=int, default=5)
    p.add_argument('--recorte', type=int, default=64, help='Tamaño de recorte CPU, no imagen completa')
    args = p.parse_args()
    if args.repeticiones < 1 or args.recorte < 1:
        p.error('repeticiones y recorte deben ser positivos')
    RESULTADOS.mkdir(exist_ok=True)
    GRAFICAS.mkdir(exist_ok=True)
    print('Sistema:', platform.platform(), '| Python:', sys.version.split()[0], '| NumPy:', np.__version__)
    if args.modo == 'gpu':
        experimentos_gpu(args.repeticiones)
    elif args.modo == 'cpu':
        experimentos_cpu(args.recorte, args.repeticiones)
    else:
        graficar()


if __name__ == '__main__':
    main()
