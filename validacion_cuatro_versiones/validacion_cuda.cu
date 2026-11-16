
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <cstdlib>
#include <cstddef>

using namespace std;
namespace fs = std::filesystem;

#define CUDA_CHECK(call) do {                              \
    cudaError_t error = (call);                            \
    if (error != cudaSuccess) {                            \
        cerr << "ERROR CUDA: "                              \
             << cudaGetErrorString(error)                   \
             << " | linea " << __LINE__ << endl;            \
        exit(EXIT_FAILURE);                                \
    }                                                       \
} while (0)

struct Filtro {
    string nombre;
    int tamano;
    vector<float> mascara;
};

Filtro crearMedia(int k) {
    return {
        "Media_" + to_string(k) + "x" + to_string(k),
        k,
        vector<float>(
            static_cast<size_t>(k) * k,
            1.0f / static_cast<float>(k * k)
        )
    };
}

Filtro crearSharpen() {
    return {
        "Sharpen_3x3",
        3,
        {
             0, -1,  0,
            -1,  5, -1,
             0, -1,  0
        }
    };
}

Filtro crearLaplaciano() {
    return {
        "Laplaciano_3x3",
        3,
        {
             0,  1,  0,
             1, -4,  1,
             0,  1,  0
        }
    };
}

// C/C++ secuencial: misma operacion de la Practica 01.
void convolucionCPU(
    const vector<float>& entrada,
    vector<float>& salida,
    int ancho,
    int alto,
    const Filtro& filtro
) {
    const int radio = filtro.tamano / 2;

    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            float suma = 0.0f;

            for (int ky = 0; ky < filtro.tamano; ++ky) {
                for (int kx = 0; kx < filtro.tamano; ++kx) {
                    const int ix = x + kx - radio;
                    const int iy = y + ky - radio;

                    if (ix >= 0 && ix < ancho &&
                        iy >= 0 && iy < alto) {
                        suma +=
                            entrada[iy * ancho + ix] *
                            filtro.mascara[
                                ky * filtro.tamano + kx
                            ];
                    }
                }
            }

            salida[y * ancho + x] = suma;
        }
    }
}

// CUDA C: un hilo por pixel, bordes en cero.
__global__ void convolucionGPU(
    const float* entrada,
    float* salida,
    int ancho,
    int alto,
    const float* mascara,
    int tamano
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    if (x >= ancho || y >= alto) return;

    const int radio = tamano / 2;
    float suma = 0.0f;

    for (int ky = 0; ky < tamano; ++ky) {
        for (int kx = 0; kx < tamano; ++kx) {
            const int ix = x + kx - radio;
            const int iy = y + ky - radio;

            if (ix >= 0 && ix < ancho &&
                iy >= 0 && iy < alto) {
                suma +=
                    entrada[iy * ancho + ix] *
                    mascara[ky * tamano + kx];
            }
        }
    }

    salida[y * ancho + x] = suma;
}

void ejecutarGPU(
    const vector<float>& entrada,
    vector<float>& salida,
    int ancho,
    int alto,
    const Filtro& filtro,
    int hilos
) {
    const size_t bytesImagen =
        entrada.size() * sizeof(float);

    const size_t bytesMascara =
        filtro.mascara.size() * sizeof(float);

    float* dEntrada = nullptr;
    float* dSalida = nullptr;
    float* dMascara = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&dEntrada),
        bytesImagen
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&dSalida),
        bytesImagen
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&dMascara),
        bytesMascara
    ));

    CUDA_CHECK(cudaMemcpy(
        dEntrada, entrada.data(),
        bytesImagen, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dMascara, filtro.mascara.data(),
        bytesMascara, cudaMemcpyHostToDevice
    ));

    dim3 bloque(hilos);
    dim3 rejilla(
        (ancho + hilos - 1) / hilos,
        alto
    );

    convolucionGPU<<<rejilla, bloque>>>(
        dEntrada,
        dSalida,
        ancho,
        alto,
        dMascara,
        filtro.tamano
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        salida.data(), dSalida,
        bytesImagen, cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(dEntrada));
    CUDA_CHECK(cudaFree(dSalida));
    CUDA_CHECK(cudaFree(dMascara));
}

// Guardar valores float32 sin recorte ni normalizacion.
void guardarBinario(
    const fs::path& ruta,
    const vector<float>& datos
) {
    ofstream archivo(ruta, ios::binary);

    if (!archivo) {
        throw runtime_error(
            "No se pudo crear: " + ruta.string()
        );
    }

    archivo.write(
        reinterpret_cast<const char*>(datos.data()),
        static_cast<streamsize>(
            datos.size() * sizeof(float)
        )
    );

    if (!archivo) {
        throw runtime_error(
            "Error al escribir: " + ruta.string()
        );
    }
}

void ejecutarCaso(
    const string& nombreCaso,
    const vector<float>& entrada,
    int ancho,
    int alto,
    const fs::path& carpeta
) {
    vector<Filtro> filtros = {
        crearMedia(3),
        crearMedia(9),
        crearMedia(21),
        crearSharpen(),
        crearLaplaciano()
    };

    for (const auto& filtro : filtros) {
        vector<float> salidaCPU(entrada.size());
        vector<float> salidaGPU(entrada.size());

        convolucionCPU(
            entrada, salidaCPU,
            ancho, alto, filtro
        );

        ejecutarGPU(
            entrada, salidaGPU,
            ancho, alto, filtro, 256
        );

        const string base =
            nombreCaso + "_" + filtro.nombre;

        guardarBinario(
            carpeta / (base + "_cpp.bin"),
            salidaCPU
        );

        guardarBinario(
            carpeta / (base + "_cuda.bin"),
            salidaGPU
        );

        cout << "Generado: " << base
             << " | CPU y CUDA C" << endl;
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            cerr << "Uso: validacion_cuda.exe "
                 << "<carpeta_salida>" << endl;
            return EXIT_FAILURE;
        }

        const fs::path carpeta = argv[1];
        fs::create_directories(carpeta);

        // Matriz manual 5x5: valores del 1 al 25.
        vector<float> matriz5(25);

        for (int i = 0; i < 25; ++i) {
            matriz5[i] = static_cast<float>(i + 1);
        }

        ejecutarCaso(
            "matriz_5x5",
            matriz5,
            5, 5,
            carpeta
        );

        // Imagen irregular: 53 columnas y 37 filas.
        vector<float> irregular(53 * 37);

        for (size_t i = 0; i < irregular.size(); ++i) {
            irregular[i] =
                static_cast<float>(i % 251);
        }

        ejecutarCaso(
            "irregular_53x37",
            irregular,
            53, 37,
            carpeta
        );

        cout << "\nRESULTADOS GENERADOS CORRECTAMENTE"
             << endl;

        return EXIT_SUCCESS;
    }
    catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
