#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "optix.h"
#include <stdint.h>

class OptixSetup {
public:
	OptixSetup();
	OptixSetup(int deviceIndex);
	~OptixSetup();

	void init();

	const CUcontext& getCudaContext() const;
	CUstream getCudaStream() const;
	cudaDeviceProp getCudaDeviceProperties() const;
	
	const OptixDeviceContext& getOptixContext() const;

protected: 
	void initOptix();
	void initContext();

protected:
	int cudaDeviceId;
	int cudaDeviceSize;

	// Cuda setup
	CUcontext cudaContext;
	CUstream cudaStream;
	cudaDeviceProp cudaDeviceProps;

	// Optix setup
	OptixDeviceContext optixContext;
};

