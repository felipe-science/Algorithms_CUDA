#include <stdio.h>
#include <time.h>

#define L 800

#define DB 200
#define DG 2



void randomizar (int matrizA[L][L] , int matriz[L][L]);
void mult_cpu (int matrizA[L][L] , int matrizB[L][L] , int matrizC[L][L]);
int soma_linha_coluna(int matrizA[L][L] , int matrizB[L][L] , int linha , int coluna);
void mult_gpu (int host_matrizA[L][L] , int host_matrizB[L][L] , int host_matrizC[L][L]);
void experimento_sequencial (int matrizA[L][L] , int matrizB[L][L] , int matrizC[L][L] , int amostras);
void experimento_paralelo (int matrizA[L][L] , int matrizB[L][L] , int matrizC[L][L] , int amostras);
void imprimir (int matriz[L][L]);


//Trecho de codigo que sera executado na placa grafica
__global__ void multiplicacao (int *matrizA , int *matrizB , int *matrizC)
{
	int lin = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

	int k;

	int soma = 0;

	if (lin < L && col < L)
	{
		soma = 0;
		for (k = 0; k < L; k++)
		{
			soma = soma + matrizA[lin*L+k]*matrizB[k*L+col];
		}

		matrizC[lin*L+col] = soma;
	}
}



int main()
{
	srand(time(NULL));

	int matrizA[L][L];
	int matrizB[L][L];
	int matrizC[L][L];

//	randomizar (matrizA , matrizB);	
//	mult_cpu   (matrizA , matrizB , matrizC);
	
//	imprimir (matrizA); printf ("\n");
//	imprimir (matrizB); printf ("\n");
//	imprimir (matrizC); printf ("\n");


	
	int amostras = 10;
	experimento_sequencial (matrizA , matrizB , matrizC , amostras);
	experimento_paralelo   (matrizA , matrizB , matrizC , amostras);


	
	return 0;
}

//Randomiza as matrizes
void randomizar (int matrizA[L][L] , int matrizB[L][L])
{
	int i, j;

	for (i  = 0; i < L; i++)
	{
		for (j = 0; j < L; j++)
		{
			matrizA[i][j] = rand() % 10;
			matrizB[i][j] = rand() % 10;
		}
	}

}

//=========================================================================================
//Calcula o produto de matriz sequencialmente
void mult_cpu (int matrizA[L][L] , int matrizB[L][L] , int matrizC[L][L])
{
	int i, j, soma;

	for (i = 0; i < L; i++)
	{
		for (j = 0; j < L; j++)
		{
			soma = soma_linha_coluna(matrizA , matrizB , i, j);
			matrizC[i][j] = soma;
		}
	}
}

//Função auxiliar de 'mult_cpu'
int soma_linha_coluna(int matrizA[L][L] , int matrizB[L][L] , int linha , int coluna)
{
	int i, soma;

	soma = 0;
	for (i = 0; i < L; i++)
	{
		soma = soma + (matrizA[linha][i]*matrizB[i][coluna]);
	}

	return soma;
}
//=============================================================================================


//Calcula o produto de matriz paralelarmente
void mult_gpu (int host_matrizA[L][L] , int host_matrizB[L][L] , int host_matrizC[L][L])
{

	int *device_matrizA;
	int *device_matrizB;
	int *device_matrizC;

	cudaMalloc ((void**) &device_matrizA , L * L * sizeof(int));
	cudaMalloc ((void**) &device_matrizB , L * L * sizeof(int));
	cudaMalloc ((void**) &device_matrizC , L * L * sizeof(int));

	cudaMemcpy (device_matrizA , host_matrizA , L * L * sizeof(int) , cudaMemcpyHostToDevice);
	cudaMemcpy (device_matrizB , host_matrizB , L * L * sizeof(int) , cudaMemcpyHostToDevice);
	cudaMemcpy (device_matrizC , host_matrizC , L * L * sizeof(int) , cudaMemcpyHostToDevice);



	dim3 DIMBLOC (DB , DB);
        dim3 DIMGRID (DG , DG);
        multiplicacao <<< DIMBLOC , DIMGRID >>> (device_matrizA , device_matrizB, device_matrizC);
	

	
	cudaMemcpy (host_matrizC , device_matrizC , L * L * sizeof(int) , cudaMemcpyDeviceToHost);

	
	cudaFree (device_matrizA);
	cudaFree (device_matrizB);
	cudaFree (device_matrizC);
	
}




void experimento_sequencial (int matrizA[L][L] , int matrizB[L][L] , int matrizC[L][L] , int amostras)
{
	int a;
	float p;

	
	time_t inicio, fim;

	inicio = time(NULL);
	for (a = 0; a < amostras; a++)
	{
		randomizar (matrizA , matrizB);
		mult_cpu   (matrizA , matrizB , matrizC);
		
		p = (100.0*a)/amostras;
		printf("%f%%\n",p);	
	}
	fim = time(NULL);

//	printf ("N_amo: %d\n",amostras);
	printf ("\n\n");
	printf ("tempo sequencial: %fs\n",difftime(fim,inicio));
	printf ("\n");
}


void experimento_paralelo (int matrizA[L][L] , int matrizB[L][L] , int matrizC[L][L] , int amostras)
{
        int a;

        time_t inicio, fim;

        inicio = time(NULL);
        for (a = 0; a < amostras; a++)
        {
        	randomizar (matrizA , matrizB);
                mult_gpu   (matrizA , matrizB , matrizC);

        }
        fim = time(NULL);

//      printf ("N_amo: %d\n",amostras);
        printf ("tempo paralelo: %fs\n",difftime(fim,inicio));
}


//Imprime uma matriz na tela
void imprimir (int matriz[L][L])
{
	int i, j;
	
	printf("\n");
	for (i = 0; i < L; i++)
	{
		for (j = 0; j < L; j++)
			printf ("%5d",matriz[i][j]);
			printf ("\n");
			printf ("\n");
			
	}
	printf("\n");
	
}

