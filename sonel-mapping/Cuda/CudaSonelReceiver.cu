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
    unsigned int index = 0;

	CudaRandom random;

	float data[(MAX_FREQUENCIES * MAX_SONELS * DATA_SIZE)];
	unsigned int dataIndices[MAX_FREQUENCIES];

	float time = 0.0f;
	float distance = 0.0f;
    unsigned int bounce = 0;

	__device__ unsigned int getDataIndex(unsigned int frequencyIndex) {
		return getSonelIndex(frequencyIndex, dataIndices[frequencyIndex]);
	}

	__device__ void addSoundSourceHit(const SimpleSoundSource& soundSource, const gdt::vec3f& rayOrigin, const gdt::vec3f& rayDirection) {
		// const float absorption[] = { 0.0012f, 0.0023f, 0.0067f, 0.0206f };
        const float absorption[] = { 0.0206, 0.0206, 0.0206, 0.0206 };
        const unsigned int dataIndex = getDataIndex(soundSource.frequencyIndex);
		if ((dataIndices[soundSource.frequencyIndex]) >= MAX_SONELS) {
			return;
		}

		gdt::vec3f soundSourceHit = getSoundSourceHitT(soundSource, params.soundSourceRadius, rayOrigin, rayDirection);
		float soundSourceDistance = gdt::length(soundSourceHit - rayOrigin) * SCALE;
		float energy = (powf(10.0f, soundSource.decibel / 10.0f) / params.rayAmount);
		float scaledEnergy = energy * exp(-absorption[soundSource.frequencyIndex] * soundSourceDistance);

		data[dataIndex + DATA_TIME_OFFSET] = (soundSourceDistance / params.timestep) + soundSource.timestamp;
		data[dataIndex + DATA_ENERGY_OFFSET] = scaledEnergy;
        data[ARRAY_HEADER_HITS_OFFSET] += 1.0f;
		dataIndices[soundSource.frequencyIndex]++;
	}

	__device__ void addSonelHit(const Sonel& sonel) {
		// const float absorption[] = { 0.0012f, 0.0023f, 0.0067f, 0.0206f };
		const float absorption[] = { 0.0206, 0.0206, 0.0206, 0.0206 };
        const unsigned int dataIndex = getDataIndex(sonel.frequencyIndex);
		if ((dataIndices[sonel.frequencyIndex]) >= MAX_SONELS) {
			return;
		}

		float totalDistance = sonel.distance + distance;
		float totalTime = sonel.time + time;
		float scale = exp(-absorption[sonel.frequencyIndex] * totalDistance);
		float scaledEnergy = sonel.energy * scale;

		// printf("Adding Sonel Hit (%fs), (%f energy) on index %d\n", totalTime, scaledEnergy, dataIndex);

		data[dataIndex + DATA_TIME_OFFSET] = totalTime;
		data[dataIndex + DATA_ENERGY_OFFSET] = scaledEnergy;
        data[ARRAY_HEADER_HITS_OFFSET] += 1.0f;
		dataIndices[sonel.frequencyIndex]++;
	}
};

extern "C" __global__ void __closesthit__radiance() {
	const float3 rayOriginOptix = optixGetWorldRayOrigin();
	const float3 rayDirectionOptix = optixGetWorldRayDirection();
	gdt::vec3f rayOrigin(rayOriginOptix.x, rayOriginOptix.y, rayOriginOptix.z);
	gdt::vec3f rayDirection(rayDirectionOptix.x, rayDirectionOptix.y, rayDirectionOptix.z);

	const SmSbtData& sbtData = *(const SmSbtData*) optixGetSbtDataPointer();
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	if (sbtData.type == SOUND_SOURCE) {
		printf("Sound Source Hit on bounce %d\n", prd.bounce);
		const SimpleSoundSource& soundSource = *(sbtData.soundSource);
		prd.addSoundSourceHit(soundSource, rayOrigin, rayDirection);
		return;
	}

	gdt::vec3f hitPosition, geometryNormal, shadingNormal;
	getSurfaceData(sbtData, hitPosition, geometryNormal, shadingNormal);
	fixNormals(rayDirection, geometryNormal, shadingNormal);

    float bounceLength = (gdt::length(rayOrigin - hitPosition) * SCALE);
	prd.distance += bounceLength;
	float duration = prd.distance / params.soundSpeed;
	prd.time += duration;

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
		newRayMin = 0.01f;
		newRayMax = 1.0f;
		newRayDir = -rayDirection;
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

    prd.bounce++;
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
	const SmSbtData& sbtData = *(const SmSbtData*) optixGetSbtDataPointer();
	const SbtDataType type = sbtData.type;
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	if (type == SONEL) {
		const Sonel& sonel = *sbtData.sonel;
		// printf("Adding Sonel\n");
		prd.addSonelHit(sonel);
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

		gdt::vec3f center = sonel.position;
		float length = gdt::length(center - rayOrigin);
		if (length < params.sonelRadius) {
			intersectionT = 0.11f;
            hit = true;
		}
	}
	else if (type == SOUND_SOURCE) {
		const SimpleSoundSource& soundSource = *(sbtData->soundSource);
        float t = getSoundSourceHitT(soundSource, soundSource.radius, rayOrigin, rayDirection);

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
	const int rayIndex = optixGetLaunchIndex().x;
	const auto& camera = params.camera;

	PerRayData prd;

	memset(prd.data, 0, sizeof(float) * getDataArraySize());
	memset(prd.dataIndices, 0, sizeof(unsigned int) * MAX_FREQUENCIES);

    prd.index = rayIndex;
	prd.random.init(rayIndex, 0, 0);
	prd.time = 0.0f;

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	// generate ray direction
	vec3f rayDir;
	prd.random.randomVec3fSphere(rayDir);

	optixTrace(
		params.traversable,
		camera.position,
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

	const unsigned int stride = getDataArraySize();
	memcpy(&(params.energies[stride * rayIndex]), prd.data, sizeof(float) * stride);
}