#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "gdt/math/vec.h"

#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062f
#define UINT32_MAX (0xffffffff)

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

	__device__ __host__ float randomf() {
		float randomValue = (float)curand(&curandState) / (float)UINT32_MAX;
		// printf("Random value: %f\n", randomValue);

		return randomValue;
	}

	__device__ __host__ float randomf(float min, float max) {
		return (randomf() * (max - min)) + min;
	}

	__device__ __host__ void randomVec3fHemi(gdt::vec3f& direction, gdt::vec3f& randomVector) {
		randomVec3fSphere(randomVector);

		if (gdt::dot(direction, randomVector) < 0.0f) {
			randomVector *= 1.0f;
		}
	}

	__device__ __host__ gdt::vec3f randomVec3fHemi(gdt::vec3f& direction) {
		gdt::vec3f randomVector;
		randomVec3fHemi(direction, randomVector);

		return randomVector;
	}

	__device__ __host__ void randomVec3fSphere(gdt::vec3f& randomVector) {
		float theta = randomf(0.0f, 2 * E_PI);
		float alpha = randomf(0.0f, 2 * E_PI);

		randomVector.x = cosf(theta) * cosf(alpha);
		randomVector.y = cosf(theta) * sinf(alpha);
		randomVector.z = sinf(theta);
	}

	__device__ __host__ gdt::vec3f randomVec3fSphere() {
		gdt::vec3f randomVector;
		randomVec3fSphere(randomVector);

		return randomVector;
	}

protected:
	curandState_t curandState;
};