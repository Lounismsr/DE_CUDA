#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "kernel.h"


__device__ float tempParticle1[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
/**
 * Runs on the GPU, called from the GPU.
*/
__device__ float fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;

    switch (SELECTED_OBJ_FUNC)  {
        case 0: 
            float y1 = 1 + (x[0] - 1)/4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS-1] - 1)/4;

            res += pow(sin(phi*y1), 2);

            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        case 1: 
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
            }
            res -= 330;
            break;
        
        case 2:
            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i+1] - 0 + 1;
                res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        case 3:
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2)/4000;
                produit *= cos(zi/pow(i+1, 0.5));
            }
            res = somme - produit + 1 - 180; 
            break;
        case 4:
            for(int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
    }

    return res;
}

/**
 * 
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void kernelUpdateParticle(float *positions, float *velocities, 
                                     float *pBests, float *gBest, float r1, 
                                     float r2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // avoid an out of bound for the array 
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS)
        return;

    //float rp = getRandomClamped();
    //float rg = getRandomClamped();
    
    float rp = r1; // random weight for personnal =>  computed from @getRandomClamped
    float rg = r2; // random weight for global =>  computed from @getRandomClamped


    // Mise à jour de velocities et positions
    velocities[i] = OMEGA * velocities[i] + 
                    c1 * rp * (pBests[i] - positions[i]) + 
                    c2 * rg * (gBest[i % NUM_OF_DIMENSIONS] - positions[i]);

    // Update posisi particle
    //Mise à jour de la position de la particule courante
    //incrémentant la position de la particule courante avec la vitesse de la particule courante
    positions[i] += velocities[i];
}

/**
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void kernelUpdatePBest(float *positions, float *pBests, float* gBest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
    {
        tempParticle1[j] = positions[i + j];
        tempParticle2[j] = pBests[i + j];
    }

    if (fitness_function(tempParticle1) < fitness_function(tempParticle2))
    {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            pBests[i + k] = positions[i + k];
    }
}


extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest)
{

    int size = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;
    
    // declare all the arrays on the device
    float *devPos;
    float *devVel;
    float *devPBest;
    float *devGBest;
    
    float temp[NUM_OF_DIMENSIONS];
        
    // Memory allocation
    cudaMalloc((void**)&devPos, sizeof(float) * size);
    cudaMalloc((void**)&devVel, sizeof(float) * size);
    cudaMalloc((void**)&devPBest, sizeof(float) * size);
    cudaMalloc((void**)&devGBest, sizeof(float) * NUM_OF_DIMENSIONS);
    
    // Thread & Block number
    int threadsNum = 32;
    int blocksNum = ceil(size / threadsNum);
    
    // Copy particle datas from host to device
    /**
     * Copy in GPU memory the data from the host 
     * */
    cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, sizeof(float) * size, 
               cudaMemcpyHostToDevice);
    cudaMemcpy(devPBest, pBests, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS,
               cudaMemcpyHostToDevice);

    // --- Instrumentation tache 1.3 : mesure ou part le temps ---
    cudaEvent_t startK1, stopK1, startK2, stopK2, startM1, stopM1, startM2, stopM2;
    cudaEventCreate(&startK1); cudaEventCreate(&stopK1);
    cudaEventCreate(&startK2); cudaEventCreate(&stopK2);
    cudaEventCreate(&startM1); cudaEventCreate(&stopM1);
    cudaEventCreate(&startM2); cudaEventCreate(&stopM2);

    float totalK1 = 0, totalK2 = 0, totalM1 = 0, totalM2 = 0, msTmp;
    double totalCPU = 0;

    clock_t loopStart = clock();

    // PSO main function
    // MAX_ITER = 30000;

    for (int iter = 0; iter < MAX_ITER; iter++)
    {

        cudaEventRecord(startK1);
        kernelUpdateParticle<<<blocksNum, threadsNum>>>(devPos, devVel,
                                                        devPBest, devGBest,
                                                        getRandomClamped(),
                                                        getRandomClamped());
        cudaEventRecord(stopK1);
        cudaEventSynchronize(stopK1);
        cudaEventElapsedTime(&msTmp, startK1, stopK1);
        totalK1 += msTmp;

        cudaEventRecord(startK2);
        kernelUpdatePBest<<<blocksNum, threadsNum>>>(devPos, devPBest,
                                                     devGBest);
        cudaEventRecord(stopK2);
        cudaEventSynchronize(stopK2);
        cudaEventElapsedTime(&msTmp, startK2, stopK2);
        totalK2 += msTmp;

        cudaEventRecord(startM1);
        cudaMemcpy(pBests, devPBest,
                   sizeof(float) * NUM_OF_PARTICLES * NUM_OF_DIMENSIONS,
                   cudaMemcpyDeviceToHost);
        cudaEventRecord(stopM1);
        cudaEventSynchronize(stopM1);
        cudaEventElapsedTime(&msTmp, startM1, stopM1);
        totalM1 += msTmp;

        clock_t cpuStart = clock();
        for(int i = 0; i < size; i += NUM_OF_DIMENSIONS)
        {
            for(int k = 0; k < NUM_OF_DIMENSIONS; k++) //ssB1
                temp[k] = pBests[i + k];

            if (host_fitness_function(temp) < host_fitness_function(gBest))
            {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    gBest[k] = temp[k];
            }
        }
        clock_t cpuEnd = clock();
        totalCPU += (double)(cpuEnd - cpuStart) * 1000.0 / CLOCKS_PER_SEC;

        cudaEventRecord(startM2);
        cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS,
                   cudaMemcpyHostToDevice);
        cudaEventRecord(stopM2);
        cudaEventSynchronize(stopM2);
        cudaEventElapsedTime(&msTmp, startM2, stopM2);
        totalM2 += msTmp;
    }

    clock_t loopEnd = clock();
    double totalLoopMs = (double)(loopEnd - loopStart) * 1000.0 / CLOCKS_PER_SEC;
    double sumParts = totalK1 + totalK2 + totalM1 + totalM2 + totalCPU;

    printf("\n--- Repartition du temps (tache 1.3, %d iterations) ---\n", MAX_ITER);
    printf("%-22s %10.2f ms  (%5.2f %%)\n", "kernelUpdateParticle", totalK1, 100.0*totalK1/totalLoopMs);
    printf("%-22s %10.2f ms  (%5.2f %%)\n", "kernelUpdatePBest",    totalK2, 100.0*totalK2/totalLoopMs);
    printf("%-22s %10.2f ms  (%5.2f %%)\n", "cudaMemcpy pBest D2H", totalM1, 100.0*totalM1/totalLoopMs);
    printf("%-22s %10.2f ms  (%5.2f %%)\n", "Boucle CPU gBest",     totalCPU, 100.0*totalCPU/totalLoopMs);
    printf("%-22s %10.2f ms  (%5.2f %%)\n", "cudaMemcpy gBest H2D", totalM2, 100.0*totalM2/totalLoopMs);
    printf("%-22s %10.2f ms\n", "Somme des parties", sumParts);
    printf("%-22s %10.2f ms\n", "Temps total boucle", totalLoopMs);
    printf("%-22s %9.2f %%\n\n", "Ecart somme/total", 100.0*fabs(sumParts-totalLoopMs)/totalLoopMs);

    cudaEventDestroy(startK1); cudaEventDestroy(stopK1);
    cudaEventDestroy(startK2); cudaEventDestroy(stopK2);
    cudaEventDestroy(startM1); cudaEventDestroy(stopM1);
    cudaEventDestroy(startM2); cudaEventDestroy(stopM2);
    // --- fin instrumentation ---

    cudaMemcpy(positions, devPos, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, devVel, sizeof(float) * size, 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(pBests, devPBest, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gBest, devGBest, sizeof(float) * NUM_OF_DIMENSIONS, 
               cudaMemcpyDeviceToHost); 
    
    
    // cleanup
    cudaFree(devPos);
    cudaFree(devVel);
    cudaFree(devPBest);
    cudaFree(devGBest);
}

