#include "cuda_runtime.h"
#include <device_launch_parameters.h>

// CUDA kernel to render a gradient image
__global__ void render(uchar4* outputData, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float r = float(x) / float(width);
        float g = float(y) / float(height);
        float b = 0.5f;

        outputData[y * width + x] = make_uchar4(
            static_cast<unsigned char>(r * 255.0f),
            static_cast<unsigned char>(g * 255.0f),
            static_cast<unsigned char>(b * 255.0f),
            255
        );
    }
}

// Host-side wrapper function to launch the CUDA kernel
extern "C" void renderCudaKernel(uchar4* outputData, int width, int height)
{
    const int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    render<<<dimGrid, dimBlock>>>(outputData, width, height);
    cudaDeviceSynchronize();
}
