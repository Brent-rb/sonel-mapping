//
// Created by brent on 27/01/2022.
//

#ifndef SONEL_MAPPING_SONELRECEIVERPARAMS_H
#define SONEL_MAPPING_SONELRECEIVERPARAMS_H

#include <optix_types.h>
#include "gdt/math/vec.h"

struct SonelReceiverParams {
	struct {
		gdt::vec3f position;
		gdt::vec3f direction;
		gdt::vec3f horizontal;
		gdt::vec3f vertical;
	} camera;

	uint32_t frequencySize;
	uint32_t timestepSize;
	uint32_t maxDepth;
	uint32_t rayAmount;

	float* energies;

	float timestep;
	float duration;
	float soundSpeed;
	float sonelRadius;
	float soundSourceRadius;
	float timeOffset;

	OptixTraversableHandle traversable;
};

#endif //SONEL_MAPPING_SONELRECEIVERPARAMS_H
