/*******************************************************************************
 *
 * t4.cu: Producto tensorial de matrices con CUDA
 *
 * Programmer: Cristobal Gallardo & Vicente Santos
 *
 * Santiago de Chile, 7/12/2023
 *
 ******************************************************************************/


#include <stdio.h>
#include <stdlib.h> 
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

#define THREADxBLOCK 1024

__global__ void Process(float *a, float *b, float *res, int r, int c, int rb, int cb) {      //Calculo del producto tensorial
    int index = blockIdx.x * blockDim.x + threadIdx.x, i, j, m, n;

    if (index < r * rb * c* cb) {
        i = (index / (cb * rb)) / c;
        j = (index / (cb * rb)) % c;
        m = (index/ cb) % rb;
        n = index % cb;
        res[index] = a[(i * c) + j] * b[(m * cb) + n];
    }
}


float *ReadMatrix(unsigned int r, unsigned int c) {            //Lee la matriz
    unsigned int i, j;
    float *mat = (float *)malloc(r * c * sizeof(float));
    for (i = 0; i < r; i = i + 1) {
        for (j = 0; j < c; j = j + 1) {
            scanf("%f", &mat[i * c + j]);
        }
    }
    return mat;
}

void PrintMatrix(unsigned int r, unsigned int c, float *mat) {       //Imprime la matriz
    unsigned int i, j;
    for (i = 0; i < r; i = i + 1) {
        for (j = 0; j < c; j = j + 1) {
            printf(" %10.2f ", mat[i * c + j]);
        }
        printf("\n");
    }
}



void Usage(char *mess) {

    printf("\nUsage: %s -M -O < data.txt\n",mess);
    printf("M = {B: procesamiento solo con bloques, T: procesamiento con bloques y hebras}\n");
    printf("O = {S: modo silencioso, V: modo vervoso}\n\n");
}

int main(int argc, char **argv){
    int m, k, n, mkkn;
    float *Matrix1, *Matrix2, *MatrixC, *Matrix1_1D, *Matrix2_1D, *MatrixC_1D, E_cpu;
    long E_wall;
    time_t  ts, te;
    clock_t cs, ce;
    ts = time(NULL);
    cs = clock();
   
   
    if (argc == 3){
        scanf("%d",&m);
        scanf("%d",&k);
        scanf("%d",&n);
        mkkn = m * k * k * n;
        printf("m = %d k = %d n = %d\n", m, k, n);
        Matrix1 = ReadMatrix(m, k);
        Matrix2 = ReadMatrix(k, n); 
        if (strcmp(argv[2], "-V") == 0) {            //Se imprimen las matrices
                printf(" Matriz A(%d,%d):\n\n", m, k); 
                PrintMatrix(m, k, Matrix1);
                printf("\n");
                printf(" Matriz B(%d,%d):\n\n", k, n);
                PrintMatrix(k, n, Matrix2);
                
        }
        cudaMalloc((void**)&Matrix1_1D, m * k * sizeof(float));             //Se asignan tamanos
        cudaMalloc((void**)&Matrix2_1D, k * n * sizeof(float));
        cudaMalloc((void**)&MatrixC_1D, m * k * k * n * sizeof(float));
        MatrixC = (float *)malloc(mkkn * sizeof(float *));
       
        cudaMemcpy(Matrix1_1D, Matrix1, m * k * sizeof(float), cudaMemcpyHostToDevice);     //Se envian al Device
        cudaMemcpy(Matrix2_1D, Matrix2, k * n * sizeof(float), cudaMemcpyHostToDevice);
        
        if (strcmp(argv[1], "-B") == 0){                          //Calculo solo bloques
            //Modo con solo Bloques
            Process<<<mkkn, 1>>>(Matrix1_1D, Matrix2_1D, MatrixC_1D, m, k, k, n);
        }
        
        if (strcmp(argv[1], "-T") == 0){                        //Calculo con hilos y bloques
            //Modo con Bloques y hebras
            Process<<<(mkkn + (THREADxBLOCK - 1)) / THREADxBLOCK, THREADxBLOCK>>>(Matrix1_1D, Matrix2_1D, MatrixC_1D, m, k, k, n);
        }
        
        cudaMemcpy(MatrixC, MatrixC_1D, m * k * k * n *sizeof(float), cudaMemcpyDeviceToHost);       //Se recibe el resultado desde el dispositivo
        
        if (strcmp(argv[2], "-V") == 0) {       //Se muestra el resultado en pantalla
            printf(" Matriz resultado(%d,%d):\n\n", m * k, k * n);
            PrintMatrix(m * k, k * n, MatrixC);
        }
        

        
    }else{
        Usage(argv[0]);
    }
    ce = clock();
    te = time(NULL);
    E_wall = (long) (te - ts);
    E_cpu = (float)(ce - cs) / CLOCKS_PER_SEC;
    printf(" Elapsed CPU Time %f Wall Time %ld \n", E_cpu, E_wall); 
    // Liberar memoria en el host
    free(Matrix1);
    free(Matrix2);
    free(MatrixC);

    // Libera memoria del Device
    cudaFree(Matrix1_1D);
    cudaFree(Matrix2_1D);
    cudaFree(MatrixC_1D);
    
        
    return 0;
}