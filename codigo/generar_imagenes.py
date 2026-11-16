
from pathlib import Path
import numpy as np
from PIL import Image

from filtros_pycuda import FiltroPyCUDA

# Rutas del proyecto
RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "imagenes" / "imagen_2000.pgm"
SALIDA = RAIZ / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)


def crear_mascaras():
    media_3 = np.ones((3, 3), dtype=np.float32) / 9.0
    media_9 = np.ones((9, 9), dtype=np.float32) / 81.0
    media_21 = np.ones((21, 21), dtype=np.float32) / 441.0

    sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    laplaciano = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    return {
        "Media_3x3": media_3,
        "Media_9x9": media_9,
        "Media_21x21": media_21,
        "Sharpen_3x3": sharpen,
        "Laplaciano_3x3": laplaciano
    }


def main():
    print("=== GENERACION DE IMAGENES PYCUDA ===")

    if not ENTRADA.exists():
        raise FileNotFoundError(f"No existe: {ENTRADA}")

    imagen = np.asarray(
        Image.open(ENTRADA).convert("L"),
        dtype=np.float32
    )

    imagen = np.ascontiguousarray(imagen)

    print(f"Imagen: {imagen.shape[1]}x{imagen.shape[0]}")

    motor = FiltroPyCUDA()
    print(f"Compilacion CUDA: {motor.compilacion_ms:.3f} ms")

    # Guardar imagen original como referencia
    Image.fromarray(imagen.astype(np.uint8)).save(
        SALIDA / "Original_2000.png"
    )

    for nombre, mascara in crear_mascaras().items():
        print(f"\nProcesando: {nombre}")

        resultado, registros = motor.ejecutar(
            imagen,
            mascara,
            hilos=256,
            repeticiones=1
        )

        # Conservar el resultado float32 antes de convertir a PNG
        np.save(SALIDA / f"{nombre}_PyCUDA_float32.npy", resultado)

        # PNG de 8 bits para visualizacion
        imagen_png = np.clip(
            resultado, 0, 255
        ).astype(np.uint8)

        ruta = SALIDA / f"{nombre}_PyCUDA.png"
        Image.fromarray(imagen_png).save(ruta)

        print(f"Guardado: {ruta.name}")
        print(f"Kernel: {registros[0]['kernel_ms']:.3f} ms")

    print("\nTODAS LAS IMAGENES GENERADAS CORRECTAMENTE")


if __name__ == "__main__":
    main()
