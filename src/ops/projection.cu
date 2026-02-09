#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

__global__ void project_lidar_to_camera_kernel(
    const float* __restrict__ points,  // (N, 3)
    const float* __restrict__ extrinsics, // (4, 4)
    const float* __restrict__ intrinsics, // (3, 3)
    float* __restrict__ uv_coords, // (N, 2)
    int num_points,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Load point
    float x = points[idx * 3 + 0];
    float y = points[idx * 3 + 1];
    float z = points[idx * 3 + 2];

    // Apply Extrinsics (World -> Camera coords)
    // R * p + t
    float xc = extrinsics[0] * x + extrinsics[1] * y + extrinsics[2] * z + extrinsics[3];
    float yc = extrinsics[4] * x + extrinsics[5] * y + extrinsics[6] * z + extrinsics[7];
    float zc = extrinsics[8] * x + extrinsics[9] * y + extrinsics[10] * z + extrinsics[11];

    // Check if point is behind camera
    if (zc <= 0) {
        uv_coords[idx * 2 + 0] = -1;
        uv_coords[idx * 2 + 1] = -1;
        return;
    }

    // Apply Intrinsics (Camera -> Image plane)
    // u = fx * (x/z) + cx
    // v = fy * (y/z) + cy
    float u = intrinsics[0] * (xc / zc) + intrinsics[2];
    float v = intrinsics[4] * (yc / zc) + intrinsics[5];

    // Check bounds
    if (u >= 0 && u < width && v >= 0 && v < height) {
        uv_coords[idx * 2 + 0] = u;
        uv_coords[idx * 2 + 1] = v;
    } else {
        uv_coords[idx * 2 + 0] = -1;
        uv_coords[idx * 2 + 1] = -1;
    }
}

void launch_projection_kernel(
    const float* points,
    const float* extrinsics,
    const float* intrinsics,
    float* uv_coords,
    int num_points,
    int width,
    int height
) {
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;

    project_lidar_to_camera_kernel<<<grid_size, block_size>>>(
        points, extrinsics, intrinsics, uv_coords, num_points, width, height
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}
