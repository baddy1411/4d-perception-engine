#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA launcher
void launch_projection_kernel(
    const float* points,
    const float* extrinsics,
    const float* intrinsics,
    float* uv_coords,
    int num_points,
    int width,
    int height
);

torch::Tensor project_lidar(
    torch::Tensor points,
    torch::Tensor extrinsics,
    torch::Tensor intrinsics,
    int width,
    int height
) {
    // Check inputs
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(extrinsics.is_cuda(), "extrinsics must be a CUDA tensor");
    TORCH_CHECK(intrinsics.is_cuda(), "intrinsics must be a CUDA tensor");

    auto num_points = points.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    auto uv_coords = torch::empty({num_points, 2}, options);

    launch_projection_kernel(
        points.data_ptr<float>(),
        extrinsics.data_ptr<float>(),
        intrinsics.data_ptr<float>(),
        uv_coords.data_ptr<float>(),
        num_points,
        width,
        height
    );

    return uv_coords;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_lidar", &project_lidar, "Project LiDAR points to camera image (CUDA)");
}
