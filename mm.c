#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define IMG_WIDTH 1200
#define IMG_HEIGHT 1200

int main(int argc, char** argv) {
    double commTime;
    MPI_Init(&argc, &argv);
    double startTimer = MPI_Wtime();
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    double realMin = -2.0;
    double realMax = 1.0;
    double imagMin = -1.5;
    double imagMax = 1.5;
    int maxIterations = 1000;
    int subRegionHeight = IMG_HEIGHT / worldSize;
    int startRow = subRegionHeight * rank;
    int endRow = startRow + subRegionHeight;
    if (rank == worldSize - 1) {
        endRow = IMG_HEIGHT;
    }

    int* row = (int*)malloc(sizeof(int) * IMG_WIDTH);
    int* imageData = (int*)malloc(sizeof(int) * IMG_WIDTH * subRegionHeight);

    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            double cReal = realMin + (realMax - realMin) * x / IMG_WIDTH;
            double cImag = imagMin + (imagMax - imagMin) * y / IMG_HEIGHT;
            double zReal = 0.0;
            double zImag = 0.0;

            int iterations = 0;
            while (zReal * zReal + zImag * zImag < 4.0 && iterations < maxIterations) {
                double nextZReal = zReal * zReal - zImag * zImag + cReal;
                double nextZImag = 2.0 * zReal * zImag + cImag;
                zReal = nextZReal;
                zImag = nextZImag;
                iterations++;
            }

            if (iterations == maxIterations) {
                row[x] = 0;
            } else {
                row[x] = iterations % 256;
            }
        }

        int rowIndex = (y - startRow) * IMG_WIDTH;

        for (int x = 0; x < IMG_WIDTH; x++) {
            imageData[rowIndex + x] = row[x];
        }
    }

    free(row);
    int* gatheredData = NULL;
    if (rank == 0) {
        gatheredData = (int*)malloc(sizeof(int) * IMG_WIDTH * IMG_HEIGHT);
    }

    double startCommTime = MPI_Wtime();
    MPI_Gather(imageData, IMG_WIDTH * subRegionHeight, MPI_INT, gatheredData, IMG_WIDTH * subRegionHeight, MPI_INT, 0, MPI_COMM_WORLD);
    double endCommTime = MPI_Wtime();
    commTime = endCommTime - startCommTime;
    free(imageData);

    if (rank == 0) {
        FILE* filePointer = fopen("mandelbrot.pgm", "wb");
        fprintf(filePointer, "P5\n%d %d\n255\n", IMG_WIDTH, IMG_HEIGHT);
        for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
            fputc(gatheredData[i], filePointer);
        }
        fclose(filePointer);
        free(gatheredData);
    }

    double endTimer = MPI_Wtime();
    double elapsedTime = endTimer - startTimer;
    printf("Elapsed time: %f seconds\n", elapsedTime);
    printf("CommTime: %f seconds\n", commTime);
    MPI_Finalize();
    return 0;
}
