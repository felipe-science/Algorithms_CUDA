#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//Trecho de código que será executado na placa gráfica
__global__ void integral (double *dados)
{

	int rank = blockIdx.x * blockDim.x + threadIdx.x;

	int k, q, N, n;
	double a, b, A, B, L, h, I;
	
	q = dados[0];
	
	//Limites de integracao
	a = 0.0;
	b = 1.0;
	n = 1000000000;

	//Intervalo que cada thread ira processar
	L = (b-a)/q;

	//Limites de integracao de cada thread
	A = rank*L;
	B = (rank+1)*L;
	N = n/q;
	h = (B-A)/N;

	//Metodo do trapezio
	I = 0.5*(4/(1+((A)*(A)))) + 0.5*(4/(1+((B)*(B))));
	for (k = 1; k < N; k++)
	{
		I = I + (4/(1+((A+k*h)*(A+k*h))));
	}
	
	//Resultado da integral de cada thread
	dados[rank] = h*I;


	
}



int main()
{
	srand(time(NULL));

	clock_t t0, tf;
	t0 = clock();


	int B, T, q, k;
	double soma, tempo;

	//Numeros de Threads e blocos
	B = 4;
	T = 1000;
	q = B*T;

	//Declaracao de um vetor que armazenarar os dados de cada thread
	double host_dados[q];
	double *device_dados;

	host_dados[0] = q;

	//Alocando memoria na placa grafica
	cudaMalloc ((void**) &device_dados , q*sizeof(double));
	
	//Copiando os dados da RAM para a placa grafica
	cudaMemcpy (device_dados, host_dados, q*sizeof(double), cudaMemcpyHostToDevice);

	//Acionamento do kernel
	integral <<< B , T >>> (device_dados);
	
	//Copiando os dados da placa grafica para a RAM
	cudaMemcpy (host_dados, device_dados, q*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree (device_dados);

	
	//Somando a contribuicao de todas as threads
	soma = 0.0;
	for (k = 0; k < q; k++)
		soma = soma + host_dados[k];
	
	printf("\n\nI = %f\n\n",soma);

	tf = clock();
	tempo = ( (double) (tf - t0) ) / CLOCKS_PER_SEC;
	printf("Tempo de execucao: %f\n",tempo);

	return 0;
}
