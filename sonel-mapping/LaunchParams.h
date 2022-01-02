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

#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"
#include "SoundSource.h"
#include "Sonel.h"

// for this simple example, we have a single ray type
enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

struct TriangleMeshSBTData {
	gdt::vec3f  color;
	gdt::vec3f* vertex;
	gdt::vec3f* normal;
	gdt::vec2f* texcoord;
	gdt::vec3i* index;

	bool hasTexture;
	cudaTextureObject_t texture;
};

struct LaunchParams {
	struct {
		uint32_t* colorBuffer;
		gdt::vec2i size;
		int accumID{ 0 };
	} frame;

	struct {
		gdt::vec3f position;
		gdt::vec3f direction;
		gdt::vec3f horizontal;
		gdt::vec3f vertical;
	} camera;

	struct {
		gdt::vec3f origin;
		gdt::vec3f du;
		gdt::vec3f dv;
		gdt::vec3f power;

	} light;

	struct {
		SoundSource soundSource;
		float echogramDuration;
		float soundSpeed;
		float earSize;

		Sonel* sonelBuffer;
		uint32_t sonelBufferSize;
		uint32_t sonelAmount;
		uint32_t sonelMaxDepth;
	} sonelMap;


	OptixTraversableHandle traversable;
};