#include <stdio.h>
#include <stdlib.h>

#define L 10


__global__ void soma (int *vetorA, int *vetorB, int *vetorC)
{

	int rank = blockIdx.x * blockDim.x + threadIdx.x;
	vetorC[rank] = vetorA[rank] + vetorB[rank];
}


int main()
{
	srand(time(NULL));

	int i;

	//Declaracoes dos vetores
	int host_vetorA[L];
	int host_vetorB[L];
	int host_vetorC[L];
	
	int *device_vetorA;
	int *device_vetorB;
	int *device_vetorC;


	for (i = 0; i < L; i++)
	{
		host_vetorA[i] = rand()%5;
		host_vetorB[i] = rand()%5;
		host_vetorC[i] = 0;
	}
	
	

	//Alocacao de memória
	cudaMalloc((void**) &device_vetorA, L*sizeof(int));
	cudaMalloc((void**) &device_vetorB, L*sizeof(int));
	cudaMalloc((void**) &device_vetorC, L*sizeof(int));

	//Copiando vetores para a placa grafica
	cudaMemcpy(device_vetorA, host_vetorA, L*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_vetorB, host_vetorB, L*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_vetorC, host_vetorC, L*sizeof(int), cudaMemcpyHostToDevice);

	//Acionando kernel
	soma <<< 1 , L  >>> (device_vetorA, device_vetorB, device_vetorC);

	//Copiando os vetores para a memoria RAM
	cudaMemcpy(host_vetorA, device_vetorA, L*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_vetorB, device_vetorB, L*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_vetorC, device_vetorC, L*sizeof(int), cudaMemcpyDeviceToHost);

	//Liberando memoria
	cudaFree(device_vetorA);
	cudaFree(device_vetorB);
	cudaFree(device_vetorC);

	
	for (i = 0; i < L; i++)
		printf("%d  +  %d  =  %d\n",host_vetorA[i], host_vetorB[i], host_vetorC[i]);	



	return 0;
}
