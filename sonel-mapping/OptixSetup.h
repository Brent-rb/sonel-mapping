#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "optix.h"
#include <stdint.h>

class OptixSetup {
public:
	OptixSetup();
	OptixSetup(uint32_t deviceIndex);
	~OptixSetup();

	void init();

	const CUcontext& getCudaContext() const;
	const CUstream getCudaStream() const;
	const cudaDeviceProp getCudaDeviceProperties() const;
	
	const OptixDeviceContext& getOptixContext() const;

protected: 
	void initOptix();
	void initContext();

protected:
	uint32_t cudaDeviceId;
	int cudaDeviceSize;

	// Cuda setup
	CUcontext cudaContext;
	CUstream cudaStream;
	cudaDeviceProp cudaDeviceProps;

	// Optix setup
	OptixDeviceContext optixContext;
};

