#pragma once

#include <cuda_runtime.h>
#include "gdt/math/vec.h"
#include "Sonel.h"

struct TriangleMeshSbtData {
	gdt::vec3f  color;
	gdt::vec3f* vertex;
	gdt::vec3f* normal;
	gdt::vec2f* texcoord;
	gdt::vec3i* index;
	Sonel* sonel;

	bool hasTexture;
	cudaTextureObject_t texture;
};
