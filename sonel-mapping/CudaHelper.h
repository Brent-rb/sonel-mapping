#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <sstream>

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1) {
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

static __forceinline__ __device__ void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPackedOptixObject() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

static __host__ void cudaCheck(cudaError_t cudaError, const char* prefix, const char* error) {
	if (cudaError != cudaSuccess) {
		std::cerr << "[" << prefix << "] CUDA Error (" << cudaError << "): " <<
			cudaGetErrorName(cudaError) << ": " << cudaGetErrorString(cudaError) << std::endl;

		std::stringstream errorMessage;
		errorMessage << "[" << prefix << "] " << error << std::endl;
		throw std::runtime_error(errorMessage.str());
	}
}

static __host__ void cudaCheck(CUresult cudaResult, const char* prefix, const char* error) {
	if (cudaResult != CUDA_SUCCESS) {
		std::cerr << "[" << prefix << "] CUDA Error (" << cudaResult << "): " << std::endl;

		std::stringstream errorMessage;
		errorMessage << "[" << prefix << "] " << error << std::endl;
		throw std::runtime_error(errorMessage.str());
	}
}

static __host__ void optixCheck(OptixResult optixError, const char* prefix, const char* error) {
	if (optixError != cudaSuccess) {
		std::cerr << "[" << prefix << "] CUDA Error (" << optixError << "): " <<
			optixGetErrorName(optixError) << ": " << optixGetErrorString(optixError) << std::endl;

		std::stringstream errorMessage;
		errorMessage << "[" << prefix << "] " << error << std::endl;
		throw std::runtime_error(errorMessage.str());
	}
}

static __host__ void cudaSyncCheck(const char* prefix, const char* error) {
	cudaError_t syncError = cudaDeviceSynchronize();

	cudaCheck(
		syncError, prefix, error
	);
}