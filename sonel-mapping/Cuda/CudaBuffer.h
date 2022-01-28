#pragma once

#include <vector>
#include <assert.h>
#include "CudaHelper.h"

class CudaBuffer {
public:
	CudaBuffer(): cudaPointer(nullptr), sizeInBytes(0) {

	}

	~CudaBuffer() {

	}

	inline CUdeviceptr getCuDevicePointer() const {
		return (CUdeviceptr)cudaPointer;
	}

	//! re-size buffer to given number of bytes
	void resize(size_t size) {
		alloc(size);
	}

	//! allocate to given number of bytes
	void alloc(size_t size) {
		tryFree();

		this->sizeInBytes = size;
		cudaCheck(
			cudaMalloc(&cudaPointer, sizeInBytes), 
			"CUDABuffer", 
			"Failed to alloc."
		);
	}

	//! free allocated memory
	void free() {
		cudaCheck(
			cudaFree(cudaPointer), 
			"CUDABuffer", 
			"Failed to free."
		);

		cudaPointer = nullptr;
		sizeInBytes = 0;
	}

	void tryFree() {
		if (cudaPointer != nullptr) {
			free();
		}
	}

	template<typename T>
	void allocAndUpload(const std::vector<T>& data) {
		alloc(data.size() * sizeof(T));
		upload((const T*)data.data(), data.size());
	}

	template<typename T>
	void upload(const T* t, size_t count) {
		assert(cudaPointer != nullptr);
		assert(sizeInBytes == count * sizeof(T));
		cudaCheck(
			cudaMemcpy(cudaPointer, reinterpret_cast<const void*>(t), count * sizeof(T), cudaMemcpyHostToDevice),
			"CUDABuffer", 
			"Failed to upload."
		);
	}

	template<typename T>
	void download(T* t, size_t count) {
		assert(cudaPointer != nullptr);
		assert(sizeInBytes == count * sizeof(T));
		cudaCheck(
			cudaMemcpy(reinterpret_cast<void*>(t), cudaPointer, count * sizeof(T), cudaMemcpyDeviceToHost),
			"CUDABuffer", 
			"Failed to download."
		);
	}

	size_t sizeInBytes = 0;
	void* cudaPointer = nullptr;
};