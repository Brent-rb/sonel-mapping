//
// Created by brent on 09/02/2022.
//

#ifndef SONEL_MAPPING_CUDASONELRECEIVERHELPER_H
#define SONEL_MAPPING_CUDASONELRECEIVERHELPER_H

#include <cuda_runtime.h>

#define MAX_FREQUENCIES 1
#define MAX_SONELS 200

#define ARRAY_HEADER_SIZE 1
#define ARRAY_HEADER_HITS_OFFSET 0

#define DATA_SIZE 2
#define DATA_TIME_OFFSET 0
#define DATA_ENERGY_OFFSET 1

const unsigned int FREQUENCY_STRIDE = MAX_SONELS * DATA_SIZE;
__device__ __host__ unsigned int getFrequencyIndex(unsigned int frequencyIndex) {
	return (frequencyIndex * FREQUENCY_STRIDE) + ARRAY_HEADER_SIZE;
}

__device__ __host__ unsigned int getSonelIndex(unsigned int frequencyIndex, unsigned int sonelIndex) {
	return getFrequencyIndex(frequencyIndex) + (sonelIndex * DATA_SIZE);
}

__device__ __host__ unsigned int getDataArraySize() {
	return getFrequencyIndex(MAX_FREQUENCIES);
}

#endif //SONEL_MAPPING_CUDASONELRECEIVERHELPER_H
