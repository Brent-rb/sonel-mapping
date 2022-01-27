#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "../SonelMapping/Models/SonelMap.h"

class CudaSonelMapperParams {
public:
	SonelMapData* sonelMapData;

	uint32_t localFrequencyIndex;
    uint32_t globalFrequencyIndex;
	uint32_t soundSourceIndex;

	OptixTraversableHandle traversable;
};