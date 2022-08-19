// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "CudaHelper.h"
#include "CudaRandom.h"
#include "CudaSonelReceiverHelper.h"
#include "CudaSceneSettings.h"
#include "CudaDeviceHelper.h"

#include "../SonelMapping/SonelVisibilityFlags.h"
#include "../SonelMapping/Models/SonelReceiverParams.h"
#include "../SonelMapping/Models/Sonel.h"
#include "../SonelMapping/Models/SmSbtData.h"
#include "../../common/gdt/gdt/math/vec.h"

using namespace gdt;

/*! launch parameters in constant memory, filled in by optix upon
	optixLaunch (this gets filled in from the buffer we pass to
	optixLaunch) */
extern "C" __constant__ SonelReceiverParams params;


/*! per-ray data now captures random number generator, so programs
	can access RNG state */
struct PerRayData {
    uint64_t index = 0;
	uint32_t frequencyIndex = 0;
	CudaRandom random;

	float distance = 0.0f;

	uint16_t* hits = nullptr;

	__device__ bool addHit(const uint16_t& frequencyIndex, const float& energy, const float& distance, const float& timeOffset, const float& distanceToHit) {
		if((*hits) == params.maxSonels) {
			return true;
		}

		float scaledEnergy = energy * exp(-params.absorptionArray[frequencyIndex] * distance);
		// printf("Energy %f scaled with absorption %f and distance %f, result %f\n", energy, params.absorptionArray[frequencyIndex], distance, scaledEnergy);

		params.entryBuffer[index].energy = scaledEnergy;
		params.entryBuffer[index].time = distance / params.soundSpeed;
		params.entryBuffer[index].distance = distanceToHit;
		index++;
		(*hits)++;

		return false;
	}

	__device__ bool addSoundSourceHit(const SimpleSoundSource& soundSource, const gdt::vec3f& rayOrigin, const gdt::vec3f& rayDirection) {
		gdt::vec3f soundSourceHit = getSoundSourceHitT(soundSource, rayOrigin, rayDirection);
		float soundSourceDistance = gdt::length(soundSourceHit - rayOrigin) * SCALE;

		float energy = powf(10.0f, soundSource.decibel / 10.0f) / params.rayAmount * powf(10.0, -3.0);
		
		return addHit(soundSource.frequencyIndex, energy, soundSourceDistance + distance, soundSource.timestamp, -1.0f);
	}

	__device__ bool addSonelHit(const Sonel& sonel, const gdt::vec3f hit) {
		return addHit(sonel.frequencyIndex, sonel.energy, sonel.distance + distance, sonel.time, gdt::length(sonel.position - hit));
	}
};

extern "C" __global__ void __closesthit__radiance() {
	const float3 rayOriginOptix = optixGetWorldRayOrigin();
	const float3 rayDirectionOptix = optixGetWorldRayDirection();
	gdt::vec3f rayOrigin(rayOriginOptix.x, rayOriginOptix.y, rayOriginOptix.z);
	gdt::vec3f rayDirection(rayDirectionOptix.x, rayDirectionOptix.y, rayDirectionOptix.z);

	const SmSbtData& sbtData = *(const SmSbtData*) optixGetSbtDataPointer();
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	if(sbtData.type == SONEL) {
		return;
	}
	else if (sbtData.type == SOUND_SOURCE) {
		const SimpleSoundSource& soundSource = *(sbtData.soundSource);
		prd.addSoundSourceHit(soundSource, rayOrigin, rayDirection);
		return;
	}

	gdt::vec3f hitPosition, geometryNormal, shadingNormal;
	getSurfaceData(sbtData, hitPosition, geometryNormal, shadingNormal);
	fixNormals(rayDirection, geometryNormal, shadingNormal);

    float bounceLength = (gdt::length(rayOrigin - hitPosition) * SCALE);
	prd.distance += bounceLength;

	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	gdt::vec3f newRayDir;
	float newRayMin = 0.0f;
	float newRayMax = 1e20f;
	unsigned int newRayMask;
	unsigned int newRayFlags = OPTIX_RAY_FLAG_NONE;

	CudaBounceType bounceType = prd.random.getBounceType(DIFFUSE_BOUNCE_PROB, SPECULAR_BOUNCE_PROB);
	if (bounceType == CudaBounceType::Diffuse) {
		// printf("Diffuse Hit\n");
		newRayMask = SONELS_VISIBLE;
		newRayMin = 0.0001f;
		newRayMax = 0.01f;
		newRayDir = shadingNormal;
	}
	else if (bounceType == CudaBounceType::Specular) {
		// printf("Specular Hit\n");
		newRayMask = GEOMETRY_VISIBLE + SOUND_SOURCES_VISIBLE;
		newRayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
		prd.random.randomVec3fHemi(shadingNormal, newRayDir);
	}
	else {
		return;
	}

	// printf("%s from(%f, %f, %f), to(%f, %f, %f)\n", newRayType, hitPosition.x, hitPosition.y, hitPosition.z, newRayDir.x, newRayDir.y, newRayDir.z);
	optixTrace(
		params.traversable,
		hitPosition,
		newRayDir,
		newRayMin,    // tmin
		newRayMax,  // tmax
		0.0f,   // rayTime
		OptixVisibilityMask(newRayMask),
		newRayFlags,
		0, 1, // SBT Offset, Stride
		0,    // missSBTIndex
		u0, u1
	);
}


extern "C" __global__ void __anyhit__radiance() {
	// printf("AnyHit\n");
	const float3 rayOriginOptix = optixGetWorldRayOrigin();
	const float3 rayDirectionOptix = optixGetWorldRayDirection();
	gdt::vec3f rayOrigin(rayOriginOptix.x, rayOriginOptix.y, rayOriginOptix.z);
	gdt::vec3f rayDirection(rayDirectionOptix.x, rayDirectionOptix.y, rayDirectionOptix.z);

	const SmSbtData& sbtData = *(const SmSbtData*) optixGetSbtDataPointer();
	const SbtDataType type = sbtData.type;
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	if (type == SONEL) {
		const Sonel& sonel = *sbtData.sonel;
		// printf("Adding Sonel\n");
		if (!prd.addSonelHit(sonel, rayOrigin));
			optixIgnoreIntersection();
	}
}

extern "C" __global__ void __intersection__radiance() {
	// printf("Intersection\n");
	const float3 rayOriginOptix = optixGetWorldRayOrigin();
	const float3 rayDirectionOptix = optixGetWorldRayDirection();
	gdt::vec3f rayOrigin(rayOriginOptix.x, rayOriginOptix.y, rayOriginOptix.z);
	gdt::vec3f rayDirection(rayDirectionOptix.x, rayDirectionOptix.y, rayDirectionOptix.z);

	PerRayData& prd = *getPackedOptixObject<PerRayData>();
	const SmSbtData* sbtData = (const SmSbtData*)optixGetSbtDataPointer();
	const SbtDataType type = sbtData->type;
	float intersectionT = -1.0f;
    bool hit = false;

	if (type == SONEL) {
		const Sonel& sonel = *(sbtData->sonel);
		if(sonel.frequencyIndex == prd.frequencyIndex) {
			gdt::vec3f center = sonel.position;
			float length = gdt::length(center - rayOrigin);
			float cosAngle = gdt::dot(rayDirection, -sonel.incidence);
			if(length < params.sonelRadius && cosAngle > -0.00001) {
				intersectionT = 0.001f;
				hit = true;
			}
			else if (cosAngle < 0) {
				// printf("Refused because cosinus: (%f, %f, %f) vs (%f, %f, %f)\n", sonel.incidence.x, sonel.incidence.y, sonel.incidence.z, rayDirection.x, rayDirection.y, rayDirection.z);
			}
		}
	}
	else if (type == SOUND_SOURCE) {
		const SimpleSoundSource& soundSource = *(sbtData->soundSource);
        float t = getSoundSourceHitT(soundSource, rayOrigin, rayDirection);

        hit = t > -0.99f;
        intersectionT = t;
	}

	if (hit) {
		optixReportIntersection(intersectionT, 0);
	}
}


//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
	PerRayData& prd = *getPackedOptixObject<PerRayData>();
	// set to constant white as background color
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
	// compute a test pattern based on pixel ID
	const uint32_t rayIndex = optixGetLaunchIndex().x;
	const uint32_t frequencyIndex = optixGetLaunchIndex().y;
	const auto& camera = params.camera;

	PerRayData prd;
    prd.index = frequencyIndex * params.maxSonels * params.rayAmount + rayIndex * params.maxSonels;
	prd.random.init(rayIndex + params.frameIndex, 0, 0);
	prd.distance = 0.0f;
	prd.hits = &(params.hitBuffer[frequencyIndex * params.rayAmount + rayIndex]);
	(*prd.hits) = 0;
	prd.frequencyIndex = frequencyIndex;

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	// generate ray direction
	vec3f rayDir, startPosition;
	prd.random.randomVec3fSphere(rayDir);
	prd.random.randomVec3fSphere(startPosition);

	startPosition = params.receiverPosition + startPosition * params.receiverRadius;

	optixTrace(
		params.traversable,
		startPosition,
		rayDir,
		0.f,    // tmin
		1e20f,  // tmax
		0.0f,   // rayTime
		OptixVisibilityMask(GEOMETRY_VISIBLE + SOUND_SOURCES_VISIBLE),
        OPTIX_RAY_FLAG_NONE,
		0,            // SBT offset
		1,               // SBT stride
		0,            // missSBTIndex
		u0, u1
	);
}