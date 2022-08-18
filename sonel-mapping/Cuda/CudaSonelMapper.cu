#include <optix_device.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "CudaHelper.h"
#include "CudaSonelMapperParams.h"
#include "CudaRandom.h"
#include "CudaSceneSettings.h"

#include "../SonelMapping/Models/Sonel.h"
#include "../SonelMapping/Models/SmSbtData.h"

extern "C" __constant__ CudaSonelMapperParams params;

class PerRayData {
public:
	__device__ __host__ PerRayData(const CudaRandom& random, const uint32_t index, const float energy):
		random(random), index(index), depth(0), distance(0.0f) {

	}

public:
	Sonel* sonels;

	CudaRandom random;

	uint32_t index;
	uint32_t depth;
	uint32_t maxDepth;

	float distance;
	float energy;
	float timeOffset;
};

extern "C" __global__ void __anyhit__sonelRadiance() {
	/* not going to be used ... */
}

extern "C" __global__ void __closesthit__sonelRadiance() {
	const SmSbtData& sbtData = *(const SmSbtData*)optixGetSbtDataPointer();
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	// ------------------------------------------------------------------
	// gather some basic hit information
	// ------------------------------------------------------------------
	const int primitiveIndex = optixGetPrimitiveIndex();
	const gdt::vec3i index = sbtData.index[primitiveIndex];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	// ------------------------------------------------------------------
	// compute normal, using either shading normal (if avail), or
	// geometry normal (fallback)
	// ------------------------------------------------------------------
	const gdt::vec3f& A = sbtData.vertex[index.x];
	const gdt::vec3f& B = sbtData.vertex[index.y];
	const gdt::vec3f& C = sbtData.vertex[index.z];
	gdt::vec3f geometryNormal = gdt::cross(B - A, C - A);
	gdt::vec3f shadingNormal = (sbtData.normal)
		? ((1.f - u - v) * sbtData.normal[index.x]
			+ u * sbtData.normal[index.y]
			+ v * sbtData.normal[index.z])
		: geometryNormal;

	// ------------------------------------------------------------------
	// face-forward and normalize normals
	// ------------------------------------------------------------------
	const gdt::vec3f rayDir = optixGetWorldRayDirection();
	const gdt::vec3f rayOrigin = optixGetWorldRayOrigin();

	if (dot(rayDir, geometryNormal) > 0.f)
        geometryNormal = -geometryNormal;
    geometryNormal = normalize(geometryNormal);

	if (dot(geometryNormal, shadingNormal) < 0.f)
        shadingNormal -= 2.f * dot(geometryNormal, shadingNormal) * geometryNormal;
    shadingNormal = normalize(shadingNormal);


	// ------------------------------------------------------------------
	// compute shadow
	// ------------------------------------------------------------------
	const gdt::vec3f surfPos
		= (1.f - u - v) * sbtData.vertex[index.x]
		+ u * sbtData.vertex[index.y]
		+ v * sbtData.vertex[index.z];

	Sonel& sonel = prd.sonels[prd.index];
	float bounceProbability = prd.random.randomF();
	if (bounceProbability > (DIFFUSE_BOUNCE_PROB + SPECULAR_BOUNCE_PROD) ||
        prd.depth + 1 == prd.maxDepth) {
        // printf("Ended ray at %d bounces, max depth %d\n", prd.depth, prd.maxDepth);
		// Absorbed
		sonel.frequency = 0;

		return;
	}

    float bounceLength = length(surfPos - rayOrigin) * SCALE;
	prd.distance += bounceLength;

    gdt::vec3f newRayDirection;
	if (bounceProbability < DIFFUSE_BOUNCE_PROB) {
        sonel.frequency = params.sonelMapData->frequencies[params.globalFrequencyIndex];
        sonel.frequencyIndex = params.globalFrequencyIndex;
		sonel.energy = prd.energy;
		sonel.position = surfPos;
		sonel.distance = prd.distance;
		sonel.time = (prd.distance / params.sonelMapData->soundSpeed) + prd.timeOffset;
		// printf("Sonel Time %f\n", sonel.time);
		sonel.incidence = rayDir;
		prd.index += 1;
		prd.depth += 1;

		prd.random.randomVec3fHemi(shadingNormal, newRayDirection);
        // printf("[Diff] Normal (%f, %f, %f), New Ray(%f, %f, %f)\n", shadingNormal.x, shadingNormal.y, shadingNormal.z, newRayDirection.x, newRayDirection.y, newRayDirection.z);
	}
	else {
		newRayDirection = (2 * dot(shadingNormal, -rayDir) * shadingNormal) + rayDir;
        // printf("[Spec] Normal (%f, %f, %f), New Ray(%f, %f, %f)\n", shadingNormal.x, shadingNormal.y, shadingNormal.z, newRayDirection.x, newRayDirection.y, newRayDirection.z);
	}



	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	optixTrace(
		params.traversable,
		surfPos,
		newRayDirection,
		1e-3f, // tmin
		1e20f, // tmax
		0.0f,  // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,            // SBT offset
		1,            // SBT stride
		0,            // missSBTIndex 
		u0, u1
	);
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__sonelRadiance() {
    // printf("Miss\n");
	PerRayData& prd = *getPackedOptixObject<PerRayData>();
	Sonel& sonel = prd.sonels[prd.index];

	sonel.frequency = 0;
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__generateSonelMap() {
	const int sonelIndex = optixGetLaunchIndex().x;
	const int decibelIndex = optixGetLaunchIndex().y;

	SoundSource& soundSource = params.sonelMapData->soundSources[params.soundSourceIndex];
	SoundFrequency& soundFrequency = soundSource.frequencies[params.localFrequencyIndex];
	const uint32_t decibelPageStride = (soundFrequency.sonelMaxDepth * soundFrequency.sonelAmount);
	const uint32_t rayIndex = decibelPageStride * decibelIndex + sonelIndex * soundFrequency.sonelMaxDepth;
	CudaRandom random = CudaRandom(rayIndex + params.frameIndex, 0, 0);
	
	float& decibels = soundFrequency.decibels[decibelIndex];
	if (decibels > -0.000001 && decibels < 0.000001) {
		return;
	}

	float energy = powf(10.0f, decibels / 10.0f) / static_cast<float>(soundFrequency.sonelAmount);
	// printf("Simulating with sonel energy: %f, %f, %d\n", energy, decibels, soundFrequency.sonelAmount);

	PerRayData prd = PerRayData(random, rayIndex, energy);
	prd.maxDepth = soundFrequency.sonelMaxDepth;
	prd.sonels = soundFrequency.sonels;
	prd.distance = 0;
	prd.timeOffset = decibelIndex * params.sonelMapData->timestep;
	prd.energy = energy;

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	gdt::vec3f rayDirection;
	prd.random.randomVec3fSphere(rayDirection);

	optixTrace(
		params.traversable,
		soundSource.position,
		rayDirection,
		0.001f, // tmin
		1e20f, // tmax
		0.0f,  // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,            // SBT offset
		1,            // SBT stride
		0,            // missSBTIndex 
		u0, u1
	);
}