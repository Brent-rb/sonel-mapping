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

#include "../SonelMapping/Models/LaunchParams.h"
#include "../SonelMapping/Models/Sonel.h"
#include "CudaHelper.h"
#include "CudaRandom.h"
#include "../SonelMapping/Models/TriangleMeshSbtData.h"
#include "../../common/gdt/gdt/math/vec.h"

using namespace gdt;

#define RADIUS 5.0f

/*! launch parameters in constant memory, filled in by optix upon
	optixLaunch (this gets filled in from the buffer we pass to
	optixLaunch) */
extern "C" __constant__ LaunchParams launchParams;

/*! per-ray data now captures random number generator, so programs
	can access RNG state */
struct PerRayData {
	CudaRandom random;

	float energies[16];
	uint32_t hits[16];
	vec3f pixelColor;
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

    uint32_t u0, u1;
    packPointer(&prd, u0, u1);

    optixTrace(
            launchParams.traversable,
            surfPos,
            -rayDirection,
            0.00001f,    // tmin
            1.0f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_NONE,
            RADIANCE_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            RADIANCE_RAY_TYPE,            // missSBTIndex
            u0, u1
    );

    // printf("[CudaSonelVisualizer] Visualising\n");

    float energy = 0;
    float frequency = 0;
    // printf("Hits: %d, energies: %f\n", prd.hits, prd.energies);

    for (int i = 0; i < launchParams.frequencySize; i++) {
        energy += prd.energies[i];
        frequency += ((i + 1) * 1.0f) / prd.energies[i];
    }
    frequency *= energy;


    // printf("[CudaSonelVisualizer] Setting pixel\n");
    if (energy < 0.001) {
        prd.pixelColor = vec3f(pixelColor.x, pixelColor.y, pixelColor.z);
    }
    else {
        prd.pixelColor = vec3f(energy * 100.0f, 0.0, frequency / (launchParams.frequencySize * 1.0f));
    }
}

extern "C" __global__ void __anyhit__radiance() {
	const TriangleMeshSbtData& sbtData = *(const TriangleMeshSbtData*) optixGetSbtDataPointer();
	PerRayData& prd = *getPackedOptixObject<PerRayData>();

	// printf("AnyHit \n");
	if (sbtData.sonel != nullptr) {
		// printf("AnyHit Sonel Frequency: %f Energy: %f\n", sbtData.sonel->frequency, sbtData.sonel->energies);
		prd.energies[sbtData.sonel->frequencyIndex] += sbtData.sonel->energy;
		prd.hits[sbtData.sonel->frequencyIndex]++;
		optixIgnoreIntersection();
	}
    else {
        optixIgnoreIntersection();
    }
}

extern "C" __global__ void __intersection__radiance() {
	const TriangleMeshSbtData* sbtData = (const TriangleMeshSbtData*)optixGetSbtDataPointer();
	const Sonel* sonelPtr = sbtData->sonel;
	
	if (sonelPtr != nullptr) {
        const float3 rayOriginOptix = optixGetWorldRayOrigin();
        const float3 rayDirectionOptix = optixGetWorldRayDirection();
        gdt::vec3f rayOrigin(rayOriginOptix.x, rayOriginOptix.y, rayOriginOptix.z);
        gdt::vec3f rayDirection(rayDirectionOptix.x, rayDirectionOptix.y, rayDirectionOptix.z);

        const gdt::vec3f center = sonelPtr->position;
        const float length = gdt::length(center - rayOrigin);
        // printf("[Intersection] Sonel(%.2f, %.2f, %.2f), Origin(%.2f, %.2f, %.2f) distance %.2f\n", center.x, center.y, center.z, rayOrigin.x, rayOrigin.y, rayOrigin.z, length);

		if (length < RADIUS) {
            optixReportIntersection(0.001f, 0);
        }
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
	prd.pixelColor = vec3f(0.1f);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
	// compute a test pattern based on pixel ID
	const int screenX = optixGetLaunchIndex().x;
	const int screenY = optixGetLaunchIndex().y;
	const int frameX = launchParams.frame.size.x;
	const auto& camera = launchParams.camera;

	const int randX = (screenX);
	const int randY = (screenY * frameX);

	curandState_t curandState;
	curand_init((randX + randY), 0, 0, &curandState);
	PerRayData prd;

    memset(prd.energies, 0, sizeof(float) * launchParams.frequencySize);
    memset(prd.hits, 0, sizeof(uint32_t) * launchParams.frequencySize);
	prd.random.init((randX + randY), 0, 0);
	prd.pixelColor = vec3f(0.f);

	// the values we store the PRD pointer in:
	uint32_t u0, u1;
	packPointer(&prd, u0, u1);


	// normalized screen plane position, in [0,1]^2
	const vec2f screen(vec2f(screenX + prd.random.randomF(), screenY + prd.random.randomF())
		/ vec2f(launchParams.frame.size));

	// generate ray direction
	vec3f rayDir = normalize(camera.direction
		+ (screen.x - 0.5f) * camera.horizontal
		+ (screen.y - 0.5f) * camera.vertical);

	optixTrace(
		launchParams.traversable,
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
		u0, u1
	);

	const int r = int(255.99f * min(prd.pixelColor.x, 1.f));
	const int g = int(255.99f * min(prd.pixelColor.y, 1.f));
	const int b = int(255.99f * min(prd.pixelColor.z, 1.f));

	// convert to 32-bit rgba value (we explicitly set alpha to 0xff
	// to make stb_image_write happy ...
	const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

	// and write to frame buffer ...
	const uint32_t fbIndex = screenX + screenY * launchParams.frame.size.x;
	launchParams.frame.colorBuffer[fbIndex] = rgba;
}