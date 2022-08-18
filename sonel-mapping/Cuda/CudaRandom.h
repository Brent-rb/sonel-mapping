#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "gdt/math/vec.h"

#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062f

class CudaRandom {
public:
	__device__ __host__ CudaRandom() {

	}

	__device__ __host__ CudaRandom(unsigned long long seed, unsigned long long subsequence, unsigned long long offset) {
		init(seed, subsequence, offset);
	}

	__device__ __host__ void init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset) {
		curand_init(seed, subsequence, offset, &curandState);
	}

	__device__ __host__ float randomF() {
		return curand_uniform(&curandState);
	}

	__device__ __host__ float normalF() {
		return curand_normal(&curandState);
	}

	__device__ __host__ float randomF(float min, float max) {
		return (randomF() * (max - min)) + min;
	}

	__device__ __host__ void randomVec3fHemi(gdt::vec3f& direction, gdt::vec3f& randomVector) {
		randomVec3fSphere(randomVector);

		if (gdt::dot(direction, randomVector) < 0.0f) {
			randomVector *= -1.0f;
		}
	}

	__device__ __host__ gdt::vec3f randomVec3fHemi(gdt::vec3f& direction) {
		gdt::vec3f randomVector;
		randomVec3fHemi(direction, randomVector);

		return randomVector;
	}

	__device__ __host__ void randomVec3fSphere(gdt::vec3f& randomVector) {
		do {
			randomVector.x = normalF();
			randomVector.y = normalF();
			randomVector.z = normalF();
		} while(gdt::length(randomVector) < 0.0001f);

		randomVector = gdt::normalize(randomVector);
	}

	__device__ __host__ gdt::vec3f randomVec3fSphere() {
		gdt::vec3f randomVector;
		randomVec3fSphere(randomVector);

		return randomVector;
	}

protected:
	curandState_t curandState;
};