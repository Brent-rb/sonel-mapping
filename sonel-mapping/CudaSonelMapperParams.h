#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "SonelMap.h"

class CudaSonelMapperParams {
public:
	SonelMapData* sonelMapData;

	uint32_t frequencyIndex;
	uint32_t soundSourceIndex;

	OptixTraversableHandle traversable;
};