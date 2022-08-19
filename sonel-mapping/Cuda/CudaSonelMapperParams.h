#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "../SonelMapping/Models/SimulationData.h"

class CudaSonelMapperParams {
public:
	SimulationData* sonelMapData;

	uint16_t localFrequencyIndex;
    uint16_t globalFrequencyIndex;
	uint32_t soundSourceIndex;

	uint64_t frameIndex;

	OptixTraversableHandle traversable;
};