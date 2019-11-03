#include<cuda_profiler_api.h>
#include<cublas_v2.h>
#include<curand.h>
#include<stdio.h>
#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#define TRAIN_CASE 55000
#define TEST_CASE 10000
#define THREAD_LEN 256
#define N_HIDDEN 8
#define N_LAYER (N_HIDDEN + 2)
#define D_INPUT 784
#define D_HIDDEN 50
#define D_OUTPUT 10
#define ETHA 0.01
#define BATCH_SIZE 256
#define EPOCH 100

__global__ void generate_interval(float *num, size_t size, float min, float max)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		num[idx] = (max-min) * num[idx] + min;
}

__global__ void initialize(float *array, size_t size, float num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		array[idx] = num;
}

__global__ void add_bias(float *h, float *bias, int dim, int batch_size)
{
	int idx = blockIdx.x + blockDim.x + threadIdx.x;
	if(idx < dim * batch_size)
		h[idx] = h[idx] + bias[idx % dim];
}

__global__ void sigmoid(float *num, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		if(num[idx] > 0) num[idx] = 1.0 / (1.0 + exp(-num[idx]));
		else num[idx] = exp(num[idx]) / (1.0 + exp(num[idx]));
	}
}

__global__ void get_max(float *h, float* max, int dim)
{
	__shared__ float result[D_OUTPUT];
	int tidx = threadIdx.x, bidx = blockIdx.x;
	result[2*tidx] = h[dim * bidx + 2*tidx];
	result[2*tidx + 1] = h[dim * bidx + 2*tidx + 1];
	__syncthreads();
	for(int n = dim/2; n != 1; n = (n+1)/2)
	{
		if(tidx + n < dim && tidx < n)
			if(result[tidx] < result[tidx+n]) result[tidx] = result[tidx+n];
		__syncthreads();
	}
	if(result[0] < result[1]) max[bidx] = result[1];
	else max[bidx] = result[0];
}

__global__ void sub_max(float *h, float *max, int dim, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		h[idx] = h[idx] - max[idx/dim];
}

__global__ void exponential(float *num, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		num[idx] = exp(num[idx]);
}

__global__ void divide_by_sum(float *h, float *sum, int dim, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		h[idx] = h[idx] / sum[idx/dim];
}

__global__ void compute_delta_y(float *y_pred, int *y_true, int ldy, size_t size, float *delta)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		int row = idx % ldy;
		int col = idx / ldy;

		if(row == y_true[col]) delta[idx] = 1.0 - y_pred[idx];
		else delta[idx] = 0.0 - y_pred[idx];
	}
}

__global__ void compute_delta_o(float *d, float *h, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		d[idx] = d[idx] * h[idx] * (1.0 - h[idx]);
}

int main()
{
	cudaSetDevice(0);
	srand(time(NULL));
	float one = 1.0, zero = 0.0, etha = ETHA;
	clock_t start, end;
	float *tfloat, *train_input, *test_input;
	int *tint, *train_label, *test_label;
	
	tfloat = (float*)malloc(sizeof(float) * D_INPUT * TRAIN_CASE);
	cudaMalloc(&train_input, sizeof(float) * D_INPUT * TRAIN_CASE);
	cudaMalloc(&test_input, sizeof(float) * D_INPUT * TEST_CASE);
	if(train_input == NULL || test_input == NULL) printf("cudamalloc fail: train_input\n");

	tint = (int*)malloc(sizeof(int) * TRAIN_CASE);
	cudaMalloc(&train_label, sizeof(int) * TRAIN_CASE);
	cudaMalloc(&test_label, sizeof(int) * TEST_CASE);
	if(train_label == NULL || test_label == NULL) printf("cudamalloc fail : train_label\n");
	
	int w_row[N_LAYER], w_col[N_LAYER], bias_dim[N_LAYER], d_row[N_LAYER], d_col, h_row[N_LAYER], h_col;
	float *w[N_LAYER], *bias[N_LAYER], *d[N_LAYER], *h[N_LAYER];
	float *one_matrix, *h_sum;
	
	d_col = h_col = BATCH_SIZE;
	for(int n = 0; n < N_LAYER; n++)
		w_row[n] = w_col[n] = bias_dim[n] = d_row[n] = h_row[n] = D_HIDDEN;
	w_col[0] = D_INPUT;
	w_row[N_LAYER-1] = bias_dim[N_LAYER-1] = d_row[N_LAYER-1] = h_row[N_LAYER-1] = D_OUTPUT;
	
	for(int n = 0; n < N_LAYER; n++)
	{
		cudaMalloc(&w[n], w_row[n] * w_col[n] * sizeof(float));
		cudaMalloc(&bias[n], bias_dim[n] * sizeof(float));
		cudaMalloc(&d[n], d_row[n] * d_col * sizeof(float));
		cudaMalloc(&h[n], h_row[n] * h_col * sizeof(float));
		if(w[n] == NULL || bias[n] == NULL || d[n] == NULL || h[n] == NULL) printf("cudamalloc fail : initialize\n");
	}

	int max_d = D_INPUT;
	if(max_d < D_HIDDEN) max_d = D_HIDDEN;
	if(max_d < D_OUTPUT) max_d = D_OUTPUT;
	if(max_d < BATCH_SIZE) max_d = BATCH_SIZE;

	cudaMalloc(&one_matrix, sizeof(float) * max_d * max_d);
	cudaMalloc(&h_sum, sizeof(float) * BATCH_SIZE);
	if(one_matrix == NULL || h_sum == NULL) printf("cudamalloc fail : my matrices");
		
	initialize<<<(max_d * max_d + THREAD_LEN)/THREAD_LEN, THREAD_LEN>>>(one_matrix, max_d * max_d, 1.0f);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock());

	for(int n = 0; n < N_LAYER; n++)
	{
		curandGenerateUniform(gen, w[n], w_row[n] * w_col[n]);
		generate_interval<<<((w_row[n] * w_col[n] + THREAD_LEN-1)/THREAD_LEN), THREAD_LEN>>>(w[n], w_row[n] * w_col[n], -1.0f, 1.0f);

		curandGenerateUniform(gen, bias[n], bias_dim[n]);
		generate_interval<<<((bias_dim[n] + THREAD_LEN-1)/THREAD_LEN), THREAD_LEN>>>(bias[n], bias_dim[n], -1.0f, 1.0f);
	}

	// load data to device
	FILE *train_image_path, *test_image_path;
	FILE *train_label_path, *test_label_path, *lab;

	train_image_path = fopen("../mnist/train_image.txt", "r");
	train_label_path = fopen("../mnist/train_label.txt", "r");
	test_image_path = fopen("../mnist/test_image.txt", "r");
	test_label_path = fopen("../mnist/test_label.txt", "r");
	lab = fopen("test_lab.txt", "w");

	//get train_input
	int buffer_size = 0;
	for(int n = 0; n < TRAIN_CASE; n++)
	{
		for(int m = 0; m < D_INPUT; m++)
		{
			fscanf(train_image_path, "%f", &tfloat[buffer_size++]);
		}
	}
	
	//get train_label
	buffer_size = 0;
	for(int n = 0; n < TRAIN_CASE; n++)
	{
		fscanf(train_label_path, "%d", &tint[buffer_size++]);
	}

	//shuffle train_input
	float fshuffle[D_INPUT];
	int ishuffle;
	for(int n = 0; n < TRAIN_CASE; n++)
	{
		int idx = rand() % (TRAIN_CASE - n) + n;
	
		ishuffle = tint[idx];
		tint[idx] = tint[n];
		tint[n] = ishuffle;
			
		memcpy(fshuffle, &tfloat[idx*D_INPUT], sizeof(float) * D_INPUT);
		memcpy(&tfloat[idx*D_INPUT], &tfloat[n*D_INPUT], sizeof(float) * D_INPUT);
		memcpy(&tfloat[n*D_INPUT], fshuffle, sizeof(float) * D_INPUT);
	}
	
	cudaMemcpy(train_input, tfloat, sizeof(float) * D_INPUT * TRAIN_CASE, cudaMemcpyHostToDevice);
	cudaMemcpy(train_label, tint, sizeof(int) * TRAIN_CASE, cudaMemcpyHostToDevice);

	//get test_input
	buffer_size = 0;
	for(int n = 0; n < TEST_CASE; n++)
	{
		for(int m = 0; m < D_INPUT; m++)
		{
			fscanf(test_image_path, "%f", &tfloat[buffer_size++]);
		}
	}

	cudaMemcpy(test_input, tfloat, sizeof(float) * D_INPUT * TEST_CASE, cudaMemcpyHostToDevice);

	//get test_label
	buffer_size = 0;
	for(int n = 0; n < TRAIN_CASE; n++)
	{
		fscanf(test_label_path, "%d", &tint[buffer_size++]);
	}

	fclose(train_image_path);
	fclose(train_label_path);
	fclose(test_image_path);
	fclose(test_label_path);	

	cudaDeviceSynchronize();
	
	//make training graph
	cudaGraph_t forwardGraph, backwardGraph;
	cudaGraphExec_t forwardExec, backwardExec;
	cudaGraphCreate(&forwardGraph, 0);
	cudaGraphCreate(&backwardGraph, 0);

	cublasHandle_t cublas;
	cublasCreate(&cublas);

	float *input;
	int *label;

	cudaMalloc(&input, sizeof(float) * D_INPUT * BATCH_SIZE);
	cudaMalloc(&label, sizeof(int) * BATCH_SIZE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cublasSetStream(cublas, stream);

	//feed forward graph
	
	int IL = 0;
	int OL = N_LAYER-1;
	cudaStreamBeginCapture(stream);

	cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, h_row[IL], h_col, w_col[IL], &one, w[IL], w_row[IL], input, w_col[IL], &zero, h[IL], h_row[IL]);
	add_bias<<<(h_row[IL] * BATCH_SIZE + THREAD_LEN -1)/THREAD_LEN, THREAD_LEN, 0 ,stream>>>(h[IL], bias[IL], bias_dim[IL], BATCH_SIZE);
	for(int layer = 1; layer <= OL; layer++)
	{
		sigmoid<<<(h_col * h_row[layer-1] + THREAD_LEN - 1)/THREAD_LEN, THREAD_LEN, 0, stream>>>(h[layer-1], h_col * h_row[layer-1]);
		cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, h_row[layer], h_col, w_col[layer], &one, w[layer], w_row[layer], h[layer-1], h_row[layer-1], &zero, h[layer], h_row[layer]);
		add_bias<<<(h_row[layer] * h_col + THREAD_LEN -1)/THREAD_LEN, THREAD_LEN, 0 ,stream>>>(h[layer], bias[layer], bias_dim[layer], BATCH_SIZE);
	}
	//subtract max to prevent overflow
	get_max<<<BATCH_SIZE, D_OUTPUT/2, 0, stream>>>(h[OL], h_sum, h_row[OL]);
	sub_max<<<(h_row[OL] * h_col + THREAD_LEN - 1)/THREAD_LEN, THREAD_LEN, 0, stream>>>(h[OL], h_sum, h_row[OL], h_row[OL] * h_col);
	//softmax
	exponential<<<(h_col * h_row[OL] + THREAD_LEN-1)/THREAD_LEN, THREAD_LEN, 0, stream>>>(h[OL], h_col * h_row[OL]);
	cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, 1, BATCH_SIZE, h_row[OL], &one, one_matrix, 1, h[OL], h_row[OL], &zero, h_sum, 1);
	divide_by_sum<<<(h_row[OL] * h_col + THREAD_LEN - 1)/THREAD_LEN, THREAD_LEN, 0, stream>>>(h[OL], h_sum, h_row[OL], h_row[OL] * h_col);

	cudaStreamEndCapture(stream, &forwardGraph);
	if(forwardGraph == NULL) printf("capture error\n");
	cudaGraphInstantiate(&forwardExec, forwardGraph, NULL, NULL, 0);

	//backward graph

	cudaStreamBeginCapture(stream);
	compute_delta_y<<<(D_OUTPUT*BATCH_SIZE+THREAD_LEN-1)/THREAD_LEN, THREAD_LEN, 0, stream>>>(h[OL], label, D_OUTPUT, D_OUTPUT*BATCH_SIZE, d[OL]);
	compute_delta_o<<<(BATCH_SIZE*h_row[OL]+THREAD_LEN-1)/THREAD_LEN, THREAD_LEN, 0, stream>>>(d[OL], h[OL], BATCH_SIZE * h_row[OL]);
	for(int layer = OL; layer > IL; layer--)
	{
		//update delta
		cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_row[layer-1], d_col, w_row[layer], &one, w[layer], w_row[layer], d[layer], d_row[layer], &zero, d[layer-1], d_row[layer-1]);
		compute_delta_o<<<(BATCH_SIZE*h_row[layer-1]+THREAD_LEN-1)/THREAD_LEN, THREAD_LEN, 0, stream>>>(d[layer-1], h[layer-1], BATCH_SIZE * h_row[layer-1]);

		//update weight and bias
		cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, w_row[layer], w_col[layer], BATCH_SIZE, &etha, d[layer], d_row[layer], h[layer-1], h_row[layer-1], &one, w[layer], w_row[layer]);
		cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, bias_dim[layer], 1, BATCH_SIZE, &etha, d[layer], d_row[layer], one_matrix, 1, &one, bias[layer], bias_dim[layer]);
	}
	cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, w_row[IL], w_col[IL], BATCH_SIZE, &etha, d[IL], d_row[IL], input, D_INPUT, &one, w[IL], w_row[IL]);
	cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, bias_dim[IL], 1, BATCH_SIZE, &etha, d[IL], d_row[IL], one_matrix, 1, &one, bias[IL], bias_dim[IL]);

	cudaStreamEndCapture(stream, &backwardGraph);
	if(backwardGraph == NULL) printf("capture error\n");
	cudaGraphInstantiate(&backwardExec, backwardGraph, NULL, NULL, 0);

	float *result;
	cudaMalloc(&result, sizeof(float) * TRAIN_CASE * D_OUTPUT);
	if(result == NULL) printf("cudamalloc fail : result\n");

	float *result_H;
	result_H = (float*)malloc(sizeof(float) * TRAIN_CASE * D_OUTPUT);

	//training (launch graph)
	start = clock();
	for(int epoch = 0; epoch < EPOCH; epoch++)
	{
		for(int n = 0; n < TRAIN_CASE/BATCH_SIZE; n++)
		{
			cudaMemcpyAsync(input, &train_input[n*BATCH_SIZE*D_INPUT], sizeof(float) * BATCH_SIZE * D_INPUT, cudaMemcpyDeviceToDevice, stream);
			cudaMemcpyAsync(label, &train_label[n*BATCH_SIZE], sizeof(int) * BATCH_SIZE, cudaMemcpyDeviceToDevice, stream);

			cudaGraphLaunch(forwardExec, stream);
			cudaGraphLaunch(backwardExec, stream);
		}

		//test
		if(epoch % 10 != 0) continue;

//		cudaMemcpy(tfloat, w[0], sizeof(float) * w_col[0] * w_row[0], cudaMemcpyDeviceToHost);
//		printf("%.1f\n", tfloat[0]);

		for(int n = 0; n < TEST_CASE; n = n + BATCH_SIZE)
		{
			cudaMemcpy(input, &test_input[n*D_INPUT], sizeof(float) * BATCH_SIZE * D_INPUT, cudaMemcpyDeviceToDevice);
			cudaGraphLaunch(forwardExec, stream);
			cudaMemcpy(&result[n*D_OUTPUT], h[N_LAYER-1], sizeof(float) * BATCH_SIZE * D_OUTPUT, cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(result_H, result, sizeof(float) * TRAIN_CASE * D_OUTPUT, cudaMemcpyDeviceToHost);

		int correct = 0, wrong = 0;
		for(int n = 0; n < TEST_CASE; n++)
		{
			float max = 0;
			int idx = 0;
			for(int m = 0; m < D_OUTPUT; m++)
			{
				if(max < result_H[n*D_OUTPUT+m])
				{
					max = result_H[n*D_OUTPUT+m];
					idx = m;
				}
//				fprintf(lab,"%f ", result_H[n*D_OUTPUT+m]);
			}
			if(idx == tint[n]) correct++;
			else wrong++;
//			fprintf(lab, "\n");
		}
		printf("%d : %d %d %f\n", epoch, correct, wrong, (float)(correct)/(float)(correct+wrong));
	}
	end = clock();

	printf("%f sec\n", (float)((end - start)/(float)CLOCKS_PER_SEC));
	printf("%d %d %f\n", EPOCH, BATCH_SIZE, ETHA);
	cudaProfilerStop();
	return 0;
}