#pragma once
#include "gdt/math/vec.h"
#include "gdt/math/box.h"
#include <map>
#include <vector>

struct Sonel {
	gdt::vec3f position;
	gdt::vec3f incidence;
	float energy;
	float time;
	uint32_t frequency;
    uint16_t frequencyIndex;
};