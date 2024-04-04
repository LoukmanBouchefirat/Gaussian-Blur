#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

#pragma once
#ifdef __INTELLISENSE__
	void __syncthreads();
#endif

int m = 32;

__global__
	void gaussian_blur(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
		int numRows, int numCols,
		const float* const filter, const int filterWidth)
{

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;
		int thread_1d_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	
	int halfWidth = filterWidth / 2;
	float value_final = 0.0f;
	for (int y = 0; y < filterWidth; y++) {
		for (int x = 0; x < filterWidth; x++) {
		int image_r = static_cast<int>(thread_2D_pos.y + (y - halfWidth));
		clamp(image_r, numRows);
		int image_c = static_cast<int>(thread_2D_pos.x + (x - halfWidth));
		clamp(image_c, numCols);

		value_final += filter[y*filterWidth + x] * static_cast<float>(inputChannel[image_r*numCols + image_c]);
		}
	}
	outputChannel[thread_1d_pos] = value_final;
	
	


}
__host__ void generateGaussian(vector<double>&, int, int);
__host__ void errCatch(cudaError_t);
template<typename T> size_t vBytes(const typename vector<T>&);

int main() {
	vector<double> hIn, hKernel, hOut;
	double* dIn, * dOut;
	int inCols, inRows;
	int kDim, kRadius;
	int outCols, outRows;
	int max = 0;
	double bw = 8;



	Mat image = imread("Colors.jpeg", IMREAD_GRAYSCALE);
	if (!image.data || !image.isContinuous()) {
		cout << "Could not open image file." << endl;
		exit(EXIT_FAILURE);
	}
	hIn.assign(image.data, image.data + image.total());
	inCols = image.cols;
	inRows = image.rows;
	hOut.resize(inCols * inRows, 0);

	
	kDim = 5; // Kernel is square and odd in dimension, should be variable at some point
	if ((inRows < 2 * kDim + 1) || (inCols < 2 * kDim + 1)) {
		cout << "Image is too small to apply kernel effectively." << endl;
		exit(EXIT_FAILURE);
	}
	kRadius = floor(kDim / 2.0); // Radius of odd kernel doesn't consider middle index
	hKernel.resize(pow(kDim, 2), 0);
	generateGaussian(hKernel, kDim, kRadius);


	outCols = inCols - (kDim-1);
	outRows = inRows - (kDim-1);



	errCatch(cudaMalloc((void**)& dIn, vBytes(hIn)));
	errCatch(cudaMemcpy(dIn, hIn.data(), vBytes(hIn), cudaMemcpyHostToDevice));
	errCatch(cudaMalloc((void**)& dOut, vBytes(hOut)));
	errCatch(cudaMemcpy(dOut, hOut.data(), vBytes(hOut), cudaMemcpyHostToDevice));
	errCatch(cudaMemcpyToSymbol(K, hKernel.data(), vBytes(hKernel)));


	int bwHalo = bw + (kDim-1); // Increase number of threads per block to account for halo cells
	dim3 dimBlock(bwHalo, bwHalo);
	dim3 dimGrid(ceil(inCols / bw), ceil(inRows / bw)); 
	Gaussian <<<dimGrid, dimBlock, bwHalo*bwHalo*sizeof(double)>>>(dIn, dOut, kDim, inCols, outCols, outRows);
	errCatch(cudaDeviceSynchronize());
	errCatch(cudaMemcpy(hOut.data(), dOut, vBytes(hOut), cudaMemcpyDeviceToHost));
	errCatch(cudaDeviceSynchronize());

	/*
	 * Normalizing output matrix values
	*/

	for (auto& value : hOut)
		max = (value > max) ? value : max;
	for (auto& value : hOut)
		value = (value * 255) / max;



	vector<int> toInt(hOut.begin(), hOut.end()); // Converting from double to integer matrix
	Mat blurImg = Mat(toInt).reshape(0, inRows);
	blurImg.convertTo(blurImg, CV_8UC1);
	Mat cropImg = blurImg(Rect(0, 0, outCols, outRows));

	
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", image);
	namedWindow("Cropped Image", WINDOW_AUTOSIZE);
	imshow("Cropped Image", cropImg);
	waitKey(0);

	image.release();
	errCatch(cudaFree(dIn));
	errCatch(cudaFree(dOut));

	exit(EXIT_SUCCESS);
}


{
  const dim3 blockSize(m, m);
  const dim3 gridSize(9, 9);

  gaussian_blur <<<gridSize, blockSize >>> (d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur <<<gridSize, blockSize >>> (d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur <<<gridSize, blockSize >>> (d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
 
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}	   


__host__
void generateGaussian(vector<double> & K, int dim, int radius) {
	double stdev = 1.0;
	double pi = 355.0 / 113.0;
	double constant = 1.0 / (2.0 * pi * pow(stdev, 2));

	for (int i = -radius; i < radius + 1; ++i)
		for (int j = -radius; j < radius + 1; ++j)
			K[(i + radius) * dim + (j + radius)] = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stdev, 2))));
}

// Catches errors returned from CUDA functions
__host__
void errCatch(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}

// Returns the size in bytes of any type of vector
template<typename T>
size_t vBytes(const typename vector<T> & v) {
	return sizeof(T)* v.size();
}