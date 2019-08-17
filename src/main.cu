#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

__global__ void copy(const unsigned char* input_image, int image_width_px, unsigned char* output_image) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    output_image[row * image_width_px + col] = 255 - input_image[row * image_width_px + col];
}


int main() {
    
    cv::Mat input_image = cv::imread("/home/user/Dropbox/Kristof-online/workspace/AimGpuAcademy/res/hip-hop.png", cv::IMREAD_COLOR);
    //cv::imshow("Input image", input_image);
    //cv::waitKey(0);
    
    unsigned char* d_inputimagedata;
    unsigned char* d_outputimagedata;
    unsigned char* h_outputimagedata;
    const int image_data_size_bytes = input_image.cols * input_image.rows * input_image.channels() * sizeof(unsigned char);
    cudaMalloc((void**)&d_inputimagedata, image_data_size_bytes);
    cudaMemcpy(d_inputimagedata, input_image.data, image_data_size_bytes, cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&d_outputimagedata, image_data_size_bytes);
    
    dim3 grid_dim(40, 135);
    dim3 block_dim(32, 16);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    const int n_kernel_runs = 1000;
    #pragma omp parallel for
    for (int i = 0; i < n_kernel_runs; ++i) {
        copy<<<grid_dim, block_dim>>>(d_inputimagedata, input_image.cols, d_outputimagedata);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: %s\n" << cudaGetErrorString(err) << std::endl;
    }
    
    float exec_time_ms = 0;
    cudaEventElapsedTime(&exec_time_ms, start, stop);
    exec_time_ms /= static_cast<float>(n_kernel_runs);
    std::cout << "Kernel execution time [ms]: " << exec_time_ms << std::endl;
    
    h_outputimagedata = (unsigned char*) malloc(image_data_size_bytes);
    cudaMemcpy(h_outputimagedata, d_outputimagedata, image_data_size_bytes, cudaMemcpyDeviceToHost); 
    cv::Mat output_image(input_image.rows, input_image.cols, CV_8UC3, h_outputimagedata);

    cudaFree(d_inputimagedata);
    cudaFree(d_outputimagedata);

    cv::imwrite("/home/user/Dropbox/Kristof-online/workspace/AimGpuAcademy/out.png", output_image);
    cv::imshow("Output image", output_image);
    cv::waitKey(0);
    
    return 0;
}