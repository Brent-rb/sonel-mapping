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

#include "../SonelMapping/SonelVisibilityFlags.h"
#include "../SonelMapping/Models/SonelReceiverParams.h"
#include "../SonelMapping/Models/Sonel.h"
#include "../SonelMapping/Models/TriangleMeshSbtData.h"
#include "../../common/gdt/gdt/math/vec.h"

using namespace gdt;

#define RADIUS 5.0f
#define MAX_FREQUENCIES 8
#define MAX_DEPTH 8


/*! launch parameters in constant memory, filled in by optix upon
	optixLaunch (this gets filled in from the buffer we pass to
	optixLaunch) */
extern "C" __constant__ SonelReceiverParams params;

/*! per-ray data now captures random number generator, so programs
	can access RNG state */
struct PerRayData {
	CudaRandom random;

	float energies[(MAX_FREQUENCIES + 1) * MAX_DEPTH];
	float time = 0.0f;
	unsigned short depth = 0;
	unsigned short energyIndex = 0;
};

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------


extern "C" __global__ void __closesthit__radiance() {
	const float3 rayOriginOptix = optixGetWorldRayOrigin();
	gdt::vec3f rayOrigin(rayOriginOptix.x, rayOriginOptix.y, rayOriginOptix.z);
	const float3 rayDirectionOptix = optixGetWorldRayDirection();
	gdt::vec3f rayDirection(rayDirectionOptix.x, rayDirectionOptix.y, rayDirectionOptix.z);

	const TriangleMeshSbtData& sbtData = *(const TriangleMeshSbtData*) optixGetSbtDataPointer();
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	// ------------------------------------------------------------------
	// gather some basic hit information
	// ------------------------------------------------------------------
	const int   primitiveIndex = optixGetPrimitiveIndex();
	const vec3i index = sbtData.index[primitiveIndex];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	// ------------------------------------------------------------------
	// compute shadow
	// ------------------------------------------------------------------
	const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x]
			  + u * sbtData.vertex[index.y]
			  + v * sbtData.vertex[index.z];

	const gdt::vec3f& A = sbtData.vertex[index.x];
	const gdt::vec3f& B = sbtData.vertex[index.y];
	const gdt::vec3f& C = sbtData.vertex[index.z];
	gdt::vec3f geometryNormal = gdt::cross(B - A, C - A);
	gdt::vec3f shadingNormal = (sbtData.normal)
	                           ? ((1.f - u - v) * sbtData.normal[index.x]
	                              + u * sbtData.normal[index.y]
	                              + v * sbtData.normal[index.z])
	                           : geometryNormal;

	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	float duration = gdt::length(rayOrigin - surfPos) / params.soundSpeed;
	prd.time += duration;

	if (prd.time > params.duration || (prd.depth + 1) == MAX_DEPTH) {
		return;
	}

	// Save the timestamp
	prd.energies[prd.energyIndex * (MAX_FREQUENCIES + 1)] = prd.time;

	optixTrace(
		params.traversable,
		surfPos,
		-rayDirection,
		0.00001f,    // tmin
		1.0f,  // tmax
		0.0f,   // rayTime
		OptixVisibilityMask(SONELS_VISIBLE),
		OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_NONE,
		0, 1, // SBT offset, SBT stride
		0, // missSBTIndex
		u0, u1
	);

	prd.depth++;
	prd.energyIndex++;

	// generate ray direction
	vec3f rayDir;
	prd.random.randomVec3fHemi(shadingNormal, rayDir);

	optixTrace(
		params.traversable,
		surfPos,
		rayDir,
		0.f,    // tmin
		1e20f,  // tmax
		0.0f,   // rayTime
		OptixVisibilityMask(GEOMETRY_VISIBLE),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,            // SBT offset
		1,               // SBT stride
		0,            // missSBTIndex
		u0, u1
	);

}

extern "C" __global__ void __anyhit__radiance() {
	const TriangleMeshSbtData& sbtData = *(const TriangleMeshSbtData*) optixGetSbtDataPointer();
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	const unsigned int energyIndex = (prd.depth * (MAX_DEPTH + 1)) + 1 + sbtData.sonel->frequencyIndex;
	// printf("AnyHit Sonel Frequency: %f Energy: %f\n", sbtData.sonel->frequency, sbtData.sonel->energies);
	prd.energies[energyIndex] += sbtData.sonel->energy;
	optixIgnoreIntersection();
}

extern "C" __global__ void __intersection__radiance() {
	const TriangleMeshSbtData* sbtData = (const TriangleMeshSbtData*)optixGetSbtDataPointer();
	const Sonel* sonelPtr = sbtData->sonel;
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	const float3 rayOriginOptix = optixGetWorldRayOrigin();
	const float3 rayDirectionOptix = optixGetWorldRayDirection();
	gdt::vec3f rayOrigin(rayOriginOptix.x, rayOriginOptix.y, rayOriginOptix.z);
	gdt::vec3f rayDirection(rayDirectionOptix.x, rayDirectionOptix.y, rayDirectionOptix.z);

	const gdt::vec3f center = sonelPtr->position;
	const float length = gdt::length(center - rayOrigin);
	const float timeDif = abs(prd.time - sonelPtr->time);
	// printf("[Intersection] Sonel(%.2f, %.2f, %.2f), Origin(%.2f, %.2f, %.2f) distance %.2f\n", center.x, center.y, center.z, rayOrigin.x, rayOrigin.y, rayOrigin.z, length);

	if (length < RADIUS && timeDif < params.timestep) {
		optixReportIntersection(0.001f, 0);
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

	memset(prd.energies, 0, sizeof(float) * (MAX_FREQUENCIES + 1) * MAX_DEPTH);
	prd.random.init(rayIndex, 0, 0);

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);


	// generate ray direction
	vec3f rayDir;
	prd.random.randomVec3fHemi(params.camera.direction, rayDir);

	optixTrace(
		params.traversable,
		camera.position,
		rayDir,
		0.f,    // tmin
		1e20f,  // tmax
		0.0f,   // rayTime
		OptixVisibilityMask(GEOMETRY_VISIBLE),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0,            // SBT offset
		1,               // SBT stride
		0,            // missSBTIndex
		u0, u1
	);

	const unsigned int stride = (MAX_FREQUENCIES + 1) * MAX_DEPTH;
	memcpy(&(params.energies[stride * rayIndex]), prd.energies, sizeof(float) * stride);
}