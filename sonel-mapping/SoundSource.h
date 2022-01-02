#pragma once
#include "gdt/math/vec.h"

struct SoundSource {
	gdt::vec3f position;
	gdt::vec3f direction;
	float decibels;
	uint32_t frequency;
};

