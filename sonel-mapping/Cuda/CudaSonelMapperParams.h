#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "../SonelMapping/Models/SonelMap.h"

class CudaSonelMapperParams {
public:
	SonelMapData* sonelMapData;

	uint16_t localFrequencyIndex;
    uint16_t globalFrequencyIndex;
	uint32_t soundSourceIndex;

	uint64_t frameIndex;

	OptixTraversableHandle traversable;
};