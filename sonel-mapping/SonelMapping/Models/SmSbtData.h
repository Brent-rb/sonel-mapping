#pragma once

#include <cuda_runtime.h>
#include "gdt/math/vec.h"
#include "Sonel.h"
#include "SimpelSoundSource.h"

enum SbtDataType {
	GEOMETRY = 0,
	SONEL,
	SOUND_SOURCE
};

struct SmSbtData {
	SbtDataType type;

	gdt::vec3f  color;
	gdt::vec3f* vertex;
	gdt::vec3f* normal;
	gdt::vec2f* texcoord;
	gdt::vec3i* index;

	bool hasTexture;
	cudaTextureObject_t texture;

	Sonel* sonel;
	SimpleSoundSource* soundSource;
};
