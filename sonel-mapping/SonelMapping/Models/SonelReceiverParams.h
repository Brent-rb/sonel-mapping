//
// Created by brent on 27/01/2022.
//

#ifndef SONEL_MAPPING_SONELRECEIVERPARAMS_H
#define SONEL_MAPPING_SONELRECEIVERPARAMS_H

#include <optix_types.h>
#include "gdt/math/vec.h"
#include "GatherEntry.h"

struct SonelReceiverParams {
	struct {
		gdt::vec3f position;
		gdt::vec3f direction;
		gdt::vec3f horizontal;
		gdt::vec3f vertical;
	} camera;

	gdt::vec3f receiverPosition;

	GatherEntry* entryBuffer;
	uint16_t* hitBuffer;
	float* absorptionArray;

	uint16_t frequencySize;
	uint32_t timestepSize;
	uint32_t rayAmount;
	uint16_t maxSonels;
	uint64_t frameIndex;

	float timestep;
	float duration;
	float soundSpeed;
	float sonelRadius;
	float receiverRadius;

	OptixTraversableHandle traversable;
};

#endif //SONEL_MAPPING_SONELRECEIVERPARAMS_H
