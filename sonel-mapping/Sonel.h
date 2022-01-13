#pragma once
#include "gdt/math/vec.h"
#include "gdt/math/box.h"
#include <map>
#include <vector>
#include "OctTree.h"

struct Sonel {
	gdt::vec3f position;
	gdt::vec3f incidence;
	float energy;
	float time;
};