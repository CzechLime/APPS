// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_grayscale (CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img) {
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y; // zdrojový obrázek
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_color_cuda_img.m_size.y) return;
    if (l_x >= t_color_cuda_img.m_size.x) return;

    // Get point from color picture
    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3 [l_y * t_color_cuda_img.m_size.x + l_x];

    int height = t_color_cuda_img.m_size.y; // výška obrázku - (height - 1 - l_y)
    int width = t_color_cuda_img.m_size.x; // šířka obrázku - (width - 1 - l_x)

    // Store BW point to new image
    t_bw_cuda_img.m_p_uchar3 [l_y * t_bw_cuda_img.m_size.x + (width - 1 - l_x)] = l_bgr;
}

void cu_run_grayscale (CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img) {
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks ((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads (l_block_size, l_block_size);
    kernel_grayscale <<<l_blocks, l_threads>>> (t_color_cuda_img, t_bw_cuda_img);

    if (( l_cerr = cudaGetLastError ()) != cudaSuccess)
        printf ("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString (l_cerr));

    cudaDeviceSynchronize ();
}

__global__ void kernel_color_level (CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img, uchar3 color) {
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y; // zdrojový obrázek
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_color_cuda_img.m_size.y) return;
    if (l_x >= t_color_cuda_img.m_size.x) return;

    // Get point from color picture
    // uchar3 l_bgr = t_color_cuda_img.m_p_uchar3 [l_y * t_color_cuda_img.m_size.x + l_x];
    uchar3 l_bgr = t_color_cuda_img.at3 (l_y, l_x);
    uchar3 finalPixel;

    if (color.x > l_bgr.x) {
        finalPixel.x = 0;
    } else finalPixel.x = l_bgr.x - color.x;
    
    if (color.y > l_bgr.y) {
        finalPixel.y = 0;
    } else finalPixel.y = l_bgr.y - color.y;

    if (color.z > l_bgr.z) {
        finalPixel.z = 0;
    } else finalPixel.z = l_bgr.z - color.z;

    // Store BW point to new image
    t_bw_cuda_img.m_p_uchar3 [l_y * t_bw_cuda_img.m_size.x + l_x].x = finalPixel.x;
    t_bw_cuda_img.m_p_uchar3 [l_y * t_bw_cuda_img.m_size.x + l_x].y = finalPixel.y;
    t_bw_cuda_img.m_p_uchar3 [l_y * t_bw_cuda_img.m_size.x + l_x].z = finalPixel.z;
}

void cu_run_color_level (CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img, uchar3 color) {
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks ((t_color_cuda_img.m_size.x + l_block_size - 1) / l_block_size, (t_color_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads (l_block_size, l_block_size);
    kernel_color_level <<<l_blocks, l_threads>>> (t_color_cuda_img, t_bw_cuda_img, color); // spustím funkci na všech vláknech (samostatně)

    if ((l_cerr = cudaGetLastError ()) != cudaSuccess)
        printf ("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString (l_cerr));

    cudaDeviceSynchronize ();
}