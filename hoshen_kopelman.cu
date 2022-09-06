#include <stdio.h>
#include <time.h>

#define L 1000
#define LB 10
#define DB 10
#define DG 10

#define TAM 20

void disparar (int host_matriz[L][L]);
void zerar (int matriz[L][L]);
void matriz_bruta (int matriz[L][L] , float p);
void imprimir (int matriz[L][L]);


__global__ void hoshen_kopelman (int *matriz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int x, y, atual, anter, super, novo;

	int proper[(LB*LB)/2];

	for (x = 0; x < (LB*LB)/2; x++)
		proper[x] = x;


	novo = 1;
	for (x = 0; x < LB; x++)
	{
		for (y = 0; y < LB; y++)
		{
			atual = (x+i*LB)*L + (y+j*LB);
			anter = atual - 1;
			super = atual - L;
			
			//Restrigindo a analise apenas a sitios ocupados
			if (matriz[atual] != 0)
			{
				//Primeiro sitio
				if (x == 0 && y == 0)
				{
					matriz[atual] = proper[novo];
					novo++;
				}

				//Primeira linha
				if (x == 0 && y != 0)
				{
					if (matriz[anter] != 0)
					{
						matriz[atual] = matriz[anter];
					}else
					{
						matriz[atual] = proper[novo];
						novo++;
					}
				}
				
				//Primeira coluna
				if (x != 0 && y == 0)
				{
					if (matriz[super] != 0)
					{
						matriz[atual] = matriz[super];
					}else
					{
						matriz[atual] = proper[novo];
						novo++;
					}
				}

				//Regiao central
				if (x != 0 && y != 0)
				{
					//Apenas sitio anterior
					if (matriz[anter] != 0 && matriz[super] == 0)
						matriz[atual] = matriz[anter];

					//Apenas sitio superior
					if (matriz[anter] == 0 && matriz[super] != 0)
						matriz[atual] = matriz[super];

					if (matriz[anter] != 0 && matriz[super] != 0)
					{
						if (matriz[anter] <= matriz[super])
						{
							proper[matriz[super]] = proper[matriz[anter]];
							matriz[atual] = proper[matriz[super]];
						}else
						{
							proper[matriz[anter]] = proper[matriz[super]];
							matriz[atual] = proper[matriz[anter]];
						}
					}
				}

				//Sem vizinhos anterior ou superior
				if (matriz[anter] == 0 && matriz[super] == 0)
				{
					matriz[atual] = proper[novo];
					novo++;
				}
				

			}

		}
	}


	
	for (x = 0; x < LB; x++)
	{
		for (y = 0; y < LB; y++)
		{
			atual = (x+i*LB)*L + (y+j*LB);
			matriz[atual] = proper[matriz[atual]];
		}
	}

	
	for (x = 0; x < LB; x++)
        {
                for (y = 0; y < LB; y++)
                {
                        atual = (x+i*LB)*L + (y+j*LB);
                        matriz[atual] = proper[matriz[atual]];
                }
        }


//	if (i == 0 && j == 0)
//	{
//		for (x = 0; x < (LB*LB)/2; x++)
//			printf ("proper[%d] = %d\n",x,proper[x]);
//	}

}

int main()
{
	srand (time(NULL));

	int matriz[L][L];
	
	zerar (matriz);
	matriz_bruta (matriz , 0.4);
	imprimir (matriz);

	disparar (matriz);
	imprimir (matriz);

	return 0;
}


void disparar (int host_matriz[L][L])
{

	int *device_matriz;

	cudaMalloc ((void**) &device_matriz , L * L * sizeof(int));
	cudaMemcpy (device_matriz , host_matriz , L * L * sizeof(int) , cudaMemcpyHostToDevice);

	dim3 DIMBLOC (DB , DB);	
	dim3 DIMGRID (DG , DG);
	hoshen_kopelman <<< DIMBLOC , DIMGRID >>> (device_matriz);

	cudaMemcpy (host_matriz , device_matriz , L * L * sizeof(int) , cudaMemcpyDeviceToHost); 

	cudaFree (device_matriz);

}



void zerar (int matriz[L][L])
{

	int i, j;

	for (i = 0; i < L; i++)
		for (j = 0; j < L; j++)
			matriz[i][j] = 0;
}


void matriz_bruta (int matriz[L][L] , float p)
{

		int i, j, aleatorio_int;
		float aleatorio_flo;

		for (i = 0; i < L; i++)
		{
			for (j = 0; j < L; j++)
			{
				aleatorio_int = (rand() % 10000)+1;
				aleatorio_flo = (1.0*aleatorio_int)/10000;

				if (aleatorio_flo <= p)
				{
					matriz[i][j] = 1;
				}else
				{
					matriz[i][j] = 0;
				}
			}
		}

}


void imprimir (int matriz[L][L])
{
	int i, j;
	int matriz_menor[TAM][TAM];

	for (i = 0; i < TAM; i++)
		for (j = 0; j < TAM; j++)
			matriz_menor[i][j] = matriz[i][j];

	printf("\n");
        for (i = 0; i < TAM; i++)
        {
                for (j = 0; j < TAM; j++)
                        printf ("%5d",matriz_menor[i][j]);
                        printf ("\n");
                        printf ("\n");
        }
        printf("\n");
}
