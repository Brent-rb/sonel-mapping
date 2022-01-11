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

#include "LaunchParams.h"
#include "gdt/random/random.h"
#include "Sonel.h"

using namespace gdt;

#define NUM_LIGHT_SAMPLES 1
#define NUM_PIXEL_SAMPLES 1
#define RADIUS 5.0f
#define UINT32_MAX (0xffffffff)
#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062f

/*! launch parameters in constant memory, filled in by optix upon
	optixLaunch (this gets filled in from the buffer we pass to
	optixLaunch) */
extern "C" __constant__ LaunchParams launchParams;

/*! per-ray data now captures random number generator, so programs
	can access RNG state */
struct PerRayData {
	curandState_t* curandState;
	vec3f pixelColor;

	uint32_t index;
	uint32_t depth;
	float distance;
	float energy;
	bool specular;

	__device__ float randomFloat() {
		float randomValue = (float)curand(curandState) / (float)UINT32_MAX;
		// printf("Random value: %f\n", randomValue);

		return randomValue;
	}

	__device__ float randomFloatBetween(float min, float max) {
		return (randomFloat() * (max - min)) + min;
	}

	__device__ void randomHemisphereVector(vec3f& direction, vec3f& randomVector) {
		randomSphereVector(randomVector);
		
		if (dot(direction, randomVector) < 0.0f) {
			randomVector *= 1.0f;
		}
	}

	__device__ void randomSphereVector(vec3f& randomVector) {
		// TODO Proper uniform random hemisphere.
		float theta = randomFloatBetween(0.0f, 2 * E_PI);
		float alpha = randomFloatBetween(0.0f, 2 * E_PI);

		randomVector.x = cosf(theta) * cosf(alpha);
		randomVector.y = cosf(theta) * sinf(alpha);
		randomVector.z = sinf(theta);
	}
};

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1) {
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

static __forceinline__ __device__ void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPerRayData() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__shadow() {
	/* not going to be used ... */
}

extern "C" __global__ void __closesthit__radiance() {
	const TriangleMeshSBTData& sbtData
		= *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
	PerRayData& prd = *getPerRayData<PerRayData>();

	// ------------------------------------------------------------------
	// gather some basic hit information
	// ------------------------------------------------------------------
	const int   primID = optixGetPrimitiveIndex();
	const vec3i index = sbtData.index[primID];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	// ------------------------------------------------------------------
	// compute diffuse material color, including diffuse texture, if
	// available
	// ------------------------------------------------------------------
	vec3f diffuseColor = sbtData.color;
	if (sbtData.hasTexture && sbtData.texcoord) {
		const vec2f tc
			= (1.f - u - v) * sbtData.texcoord[index.x]
			+ u * sbtData.texcoord[index.y]
			+ v * sbtData.texcoord[index.z];

		vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
		diffuseColor *= (vec3f)fromTexture;
	}

	// start with some ambient term
	vec3f pixelColor = diffuseColor;

	// ------------------------------------------------------------------
	// compute shadow
	// ------------------------------------------------------------------
	const vec3f surfPos
		= (1.f - u - v) * sbtData.vertex[index.x]
		+ u * sbtData.vertex[index.y]
		+ v * sbtData.vertex[index.z];
	
	float energy = 0.0f;
	OctTreeResult<Sonel> octTreeHit = launchParams.octTree->get(surfPos);

	for (int i = 0; i < octTreeHit.currentItems; i++) {
		Sonel* sonel = &(octTreeHit.data[i].data);
		
		float distance = length(surfPos - sonel->position);
		if (distance < RADIUS) {
			energy += sonel->energy;
		}
	}
	
	if (energy < 0.0001f) {
		prd.pixelColor = vec3f(pixelColor.x, pixelColor.x, pixelColor.x);
	}
	else {
		prd.pixelColor = vec3f(round(energy) / 255, (int)(round(energy)) % 255, 0.0f);
	}
	
}

extern "C" __global__ void __anyhit__radiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __anyhit__shadow() { /*! not going to be used */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
	PerRayData& prd = *getPerRayData<PerRayData>();
	// set to constant white as background color
	prd.pixelColor = vec3f(1.f);
}

extern "C" __global__ void __miss__shadow() {
	// we didn't hit anything, so the light is visible
	vec3f& prd = *(vec3f*)getPerRayData<vec3f>();
	prd = vec3f(1.f);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
	// compute a test pattern based on pixel ID
	const int screenX = optixGetLaunchIndex().x;
	const int screenY = optixGetLaunchIndex().y;
	const int frameX = launchParams.frame.size.x;
	const int frameY = launchParams.frame.size.y;
	const int accumId = launchParams.frame.accumID;
	const auto& camera = launchParams.camera;

	const int randX = (screenX);
	const int randY = (screenY * frameX);

	curandState_t curandState;
	curand_init((randX + randY) * accumId, 0, 0, &curandState);
	PerRayData prd;
	prd.curandState = &curandState;
	prd.pixelColor = vec3f(0.f);

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);


	vec3f pixelColor = 0.f;

	// normalized screen plane position, in [0,1]^2
	const vec2f screen(vec2f(screenX + prd.randomFloat(), screenY + prd.randomFloat())
		/ vec2f(launchParams.frame.size));

	// generate ray direction
	vec3f rayDir = normalize(camera.direction
		+ (screen.x - 0.5f) * camera.horizontal
		+ (screen.y - 0.5f) * camera.vertical);

	optixTrace(launchParams.traversable,
		camera.position,
		rayDir,
		0.f,    // tmin
		1e20f,  // tmax
		0.0f,   // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
		RADIANCE_RAY_TYPE,            // SBT offset
		RAY_TYPE_COUNT,               // SBT stride
		RADIANCE_RAY_TYPE,            // missSBTIndex 
		u0, u1);
	pixelColor += prd.pixelColor;

	const int r = int(255.99f * min(pixelColor.x, 1.f));
	const int g = int(255.99f * min(pixelColor.y, 1.f));
	const int b = int(255.99f * min(pixelColor.z, 1.f));

	// convert to 32-bit rgba value (we explicitly set alpha to 0xff
	// to make stb_image_write happy ...
	const uint32_t rgba = 0xff000000
		| (r << 0) | (g << 8) | (b << 16);

	// and write to frame buffer ...
	const uint32_t fbIndex = screenX + screenY * launchParams.frame.size.x;
	launchParams.frame.colorBuffer[fbIndex] = rgba;
}














//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__sonelShadow() {
	/* not going to be used ... */
}

extern "C" __global__ void __closesthit__sonelRadiance() {
	const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
	PerRayData& prd = *getPerRayData<PerRayData>();

	// ------------------------------------------------------------------
	// gather some basic hit information
	// ------------------------------------------------------------------
	const int primID = optixGetPrimitiveIndex();
	const vec3i index = sbtData.index[primID];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	// ------------------------------------------------------------------
	// compute normal, using either shading normal (if avail), or
	// geometry normal (fallback)
	// ------------------------------------------------------------------
	const vec3f& A = sbtData.vertex[index.x];
	const vec3f& B = sbtData.vertex[index.y];
	const vec3f& C = sbtData.vertex[index.z];
	vec3f Ng = cross(B - A, C - A);
	vec3f Ns = (sbtData.normal)
		? ((1.f - u - v) * sbtData.normal[index.x]
			+ u * sbtData.normal[index.y]
			+ v * sbtData.normal[index.z])
		: Ng;

	// ------------------------------------------------------------------
	// face-forward and normalize normals
	// ------------------------------------------------------------------
	const vec3f rayDir = optixGetWorldRayDirection();
	const vec3f rayOrigin = optixGetWorldRayOrigin();

	if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
	Ng = normalize(Ng);

	if (dot(Ng, Ns) < 0.f)
		Ns -= 2.f * dot(Ng, Ns) * Ng;
	Ns = normalize(Ns);


	// ------------------------------------------------------------------
	// compute shadow
	// ------------------------------------------------------------------
	const vec3f surfPos
		= (1.f - u - v) * sbtData.vertex[index.x]
		+ u * sbtData.vertex[index.y]
		+ v * sbtData.vertex[index.z];

	uint32_t sonelIndex = (prd.index * launchParams.sonelMap.sonelMaxDepth) + prd.depth;

	Sonel* sonel = &launchParams.sonelMap.sonelBuffer[sonelIndex];
	float bounceProbality = prd.randomFloat();
	if (bounceProbality > 0.90f || prd.depth + 1 == launchParams.sonelMap.sonelMaxDepth) {
		// Absorbed
		sonel->distance = 0;
		sonel->energy = 0;
		return;
	}

	vec3f newRayDirection;
	prd.distance += length(surfPos - rayOrigin);

	if (bounceProbality < 0.5f) {
		prd.depth += 1;
		sonel->energy = prd.energy;
		sonel->frequency = launchParams.sonelMap.soundSource.frequency;
		sonel->position = surfPos;
		sonel->distance = prd.distance;
		sonel->incidence = rayDir;

		prd.randomHemisphereVector(Ns, newRayDirection);
	}
	else {
		newRayDirection = Ns * 2 * dot(Ns, rayDir) - rayDir;
	}

	

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	optixTrace(
		launchParams.traversable,
		surfPos,
		newRayDirection,
		1e-3f,      // tmin
		1e20f,  // tmax
		0.0f,       // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
		RADIANCE_RAY_TYPE,            // SBT offset
		RAY_TYPE_COUNT,               // SBT stride
		RADIANCE_RAY_TYPE,            // missSBTIndex 
		u0, u1
	);
}

extern "C" __global__ void __anyhit__sonelRadiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __anyhit__sonelShadow() { /*! not going to be used */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__sonelRadiance() {
	PerRayData& prd = *getPerRayData<PerRayData>();

	uint32_t sonelIndex = (prd.index * launchParams.sonelMap.sonelMaxDepth) + prd.depth;

	Sonel* sonel = &launchParams.sonelMap.sonelBuffer[sonelIndex];

	sonel->distance = 0;
	sonel->energy = 0;
}

extern "C" __global__ void __miss__sonelShadow() {
	// we didn't hit anything, so the light is visible
	vec3f& prd = *(vec3f*)getPerRayData<vec3f>();
	prd = vec3f(1.f);
}


//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__generateSonelMap() {
	// compute a test pattern based on pixel ID
	const int sonelIndex = optixGetLaunchIndex().x;
	const int sonelAmount = launchParams.frame.size.x;
	const auto& camera = launchParams.camera;

	curandState_t curandState;
	curand_init(sonelIndex, 0, 0, &curandState);

	PerRayData prd;
	prd.curandState = &curandState;
	prd.pixelColor = vec3f(0.f);
	prd.index = sonelIndex;
	prd.depth = 0;
	prd.distance = 0;
	prd.specular = false;
	prd.energy = launchParams.sonelMap.soundSource.decibels / sonelAmount;

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);

	vec3f pixelColor = 0.f;
	vec3f rayDirection;

	prd.randomHemisphereVector(launchParams.sonelMap.soundSource.direction, rayDirection);

	optixTrace(
		launchParams.traversable,
		launchParams.sonelMap.soundSource.position,
		rayDirection,
		0.001f, // tmin
		1e20f, // tmax
		0.0f,  // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
		RADIANCE_RAY_TYPE,            // SBT offset
		RAY_TYPE_COUNT,               // SBT stride
		RADIANCE_RAY_TYPE,            // missSBTIndex 
		u0, u1
	);
}
