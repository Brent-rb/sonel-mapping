#include <iostream>
#include <optix_stubs.h>

#include "OptixSetup.h"
#include "../Cuda/CudaHelper.h"

static void optixCallback(unsigned int level, const char* tag, const char* message, void*) {
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

OptixSetup::OptixSetup(): OptixSetup(0) {

}

OptixSetup::OptixSetup(int deviceIndex): cudaDeviceId(deviceIndex) {
	init();
}

OptixSetup::~OptixSetup() {
}

void OptixSetup::init() {
	initOptix();
	initContext();
}

const CUcontext& OptixSetup::getCudaContext() const {
	return cudaContext;
}

CUstream OptixSetup::getCudaStream() const {
	return cudaStream;
}

cudaDeviceProp OptixSetup::getCudaDeviceProperties() const {
	return cudaDeviceProps;
}

const OptixDeviceContext& OptixSetup::getOptixContext() const {
	return optixContext;
}

void OptixSetup::initOptix() {
	std::cout << "[OptixSetup] Initializing Optix..." << std::endl;

	cudaGetDeviceCount(&cudaDeviceSize);
	if (cudaDeviceSize == 0)
		throw std::runtime_error("[OptixSetup] No CUDA devices found.");

	std::cout << "[OptixSetup] Found " << cudaDeviceSize << " CUDA devices." << std::endl;

	OptixResult result = optixInit();
	if (result != OPTIX_SUCCESS) {
		std::cerr << "[OptixSetup] Optix initialization failed with code " << result << std::endl;
		std::cerr << "/t" << optixGetErrorName(result) << ": " << optixGetErrorString(result) << std::endl;
	}

	std::cout << "[OptixSetup] Optix initialized." << std::endl;
}

void OptixSetup::initContext() {
	std::cout << "[OptixSetup] Initializing CUDA context..." << std::endl;

	if (cudaDeviceId >= static_cast<uint32_t>(cudaDeviceSize)) {
		throw std::runtime_error("[OptixSetup] CUDA device id is greater than the amount of CUDA devices.");
	}

	cudaCheck(
	    cudaSetDevice(cudaDeviceId),
	    "OptixSetup",
	    "Failed to set device id."
    );

	cudaCheck(
    	cudaStreamCreate(&cudaStream),
    	"OptixSetup",
    	"Failed to create CUDA stream."
    );

	cudaCheck(
		cudaGetDeviceProperties(&cudaDeviceProps, cudaDeviceId),
		"OptixSetup",
		"Failed to retrieve device properties."
	);

	cudaCheck(
		cuCtxGetCurrent(&cudaContext),
		"OptixSetup",
		"Failed to retrieve CUDA context."
	);

	optixCheck(
		optixDeviceContextCreate(cudaContext, nullptr, &optixContext),
		"OptixSetup",
		"Failed to create OptiX context."
	);

	optixCheck(
		optixDeviceContextSetLogCallback(optixContext, optixCallback, nullptr, 4),
		"OptixSetup",
		"Failed to set OptiX log callback."
	);
}
