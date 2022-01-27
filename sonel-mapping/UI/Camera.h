#pragma once
#include "gdt/math/vec.h"

struct Camera {
	/*! camera position - *from* where we are looking */
	gdt::vec3f from;
	/*! which point we are looking *at* */
	gdt::vec3f at;
	/*! general up-vector */
	gdt::vec3f up;
};