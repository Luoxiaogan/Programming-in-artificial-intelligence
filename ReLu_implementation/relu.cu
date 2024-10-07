#include <iostream>
#include <cuda_runtime.h>

const int N = 64;
const int size = N * sizeof(float);

__global__ void relu_gpu(float *d_in, float *d_out, int N)
{
    int idx = threadIdx.x; // 每一个线程的索引
    if (idx < N)
    {
        d_out[idx] = d_in[idx] > 0 ? d_in[idx] : 0;
    }
}

int main()
{
    float *h_in = (float *)malloc(size); // 给host(cpu)上分配float数组的内存空间
    float *h_out = (float *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        h_in[i] = i;
    }

    // 1.在GPU上分配内存
    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_in, size); // 分配的GPU内存的指针会存储在d_in中，但为了修改它，需要传递指针的指针
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    relu_gpu<<<1, N>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        std::cout << h_out[i] << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    free(h_out);
}