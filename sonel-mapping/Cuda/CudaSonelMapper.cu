#include <optix_device.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../SonelMapping/Models/Sonel.h"
#include "CudaHelper.h"
#include "CudaSonelMapperParams.h"
#include "CudaRandom.h"
#include "../SonelMapping/Models/TriangleMeshSbtData.h"

#define DIFFUSE_BOUNCE_PROB 0.45f
#define SPECULAR_BOUNCE_PROD 0.45f

extern "C" __constant__ CudaSonelMapperParams params;

class PerRayData {
public:
	__device__ __host__ PerRayData(const CudaRandom& random, const uint32_t index, const float energy):
            random(random), index(index), dataDepth(0), depth(0), distance(0.0f), energies(energy), specularBounce(false) {

	}

public:
	Sonel* sonels;

	CudaRandom random;

	uint32_t index;
	uint32_t dataDepth;
	uint32_t depth;
	uint32_t maxDepth;

	float distance;
	float energies;

	bool specularBounce;
};

extern "C" __global__ void __anyhit__sonelRadiance() {
	/* not going to be used ... */
}

extern "C" __global__ void __closesthit__sonelRadiance() {
	const TriangleMeshSbtData& sbtData = *(const TriangleMeshSbtData*)optixGetSbtDataPointer();
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

	uint32_t sonelIndex = (prd.index * prd.maxDepth) + prd.dataDepth;

	Sonel& sonel = prd.sonels[sonelIndex];
	float bounceProbability = prd.random.randomF();
	if (bounceProbability > (DIFFUSE_BOUNCE_PROB + SPECULAR_BOUNCE_PROD) ||
        prd.depth + 1 == prd.maxDepth) {
		// Absorbed
		sonel.time = 0;
		sonel.energy = 0;
		return;
	}

	gdt::vec3f newRayDirection;
	prd.distance += length(surfPos - rayOrigin);

	prd.depth += 1;
	if (bounceProbability < DIFFUSE_BOUNCE_PROB) {
		prd.dataDepth += 1;
        sonel.frequency = params.sonelMapData->frequencies[params.globalFrequencyIndex];
        sonel.frequencyIndex = params.globalFrequencyIndex;
		sonel.energy = prd.energies;
		sonel.position = surfPos;
		sonel.time = prd.distance / params.sonelMapData->soundSpeed;
		sonel.incidence = rayDir;

		prd.random.randomVec3fHemi(shadingNormal, newRayDirection);
	}
	else {
		newRayDirection = shadingNormal * 2 * dot(shadingNormal, rayDir) - rayDir;
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
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	uint32_t sonelIndex = (prd.index * prd.maxDepth) + prd.dataDepth;

	Sonel& sonel = prd.sonels[sonelIndex];

	sonel.time = 0;
	sonel.energy = 0;
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__generateSonelMap() {
	const int sonelIndex = optixGetLaunchIndex().x;
	const int decibelIndex = optixGetLaunchIndex().y;
	CudaRandom random = CudaRandom(sonelIndex, 0, 0);

	SoundSource& soundSource = params.sonelMapData->soundSources[params.soundSourceIndex];
	SoundFrequency& soundFrequency = soundSource.frequencies[params.localFrequencyIndex];

	
	float& decibels = soundFrequency.decibels[decibelIndex];
	if (decibels > -0.000001 && decibels < 0.000001) {
		return;
	}

	float energy = (powf(10, decibels / 10) / soundFrequency.sonelAmount) * 1e-12f;

	PerRayData prd = PerRayData(random, sonelIndex, energy);
	prd.maxDepth = soundFrequency.sonelMaxDepth;
	prd.sonels = soundFrequency.sonels;
	prd.distance = decibelIndex * params.sonelMapData->timestep * params.sonelMapData->soundSpeed;

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	gdt::vec3f rayDirection;

	prd.random.randomVec3fHemi(soundSource.direction, rayDirection);

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