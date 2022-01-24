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
#include "OctTree.h"
#include "Sonel.h"

// for this simple example, we have a single ray type
enum { RADIANCE_RAY_TYPE = 0, RAY_TYPE_COUNT };

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

	OctTree<Sonel>* octTree;
	OptixTraversableHandle traversable;
};