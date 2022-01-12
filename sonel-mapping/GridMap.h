#pragma once
#include "Sonel.h"
#include "gdt/math/box.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

class GridSonelPage {
public:
	Sonel* sonels;
	uint32_t size;
};

class GridSonelMap {
public:
	GridSonelMap();
	GridSonelMap(const std::vector<Sonel>& sonels, const gdt::box3f bounds, const float resolution);
	~GridSonelMap();

	void parse(const std::vector<Sonel>& sonels, const gdt::box3f bounds, const float resolution);

	__device__ __host__ GridSonelPage& get(gdt::vec3f& position);

	GridSonelMap* upload();

protected:
	float resolution;
	gdt::box3f bounds;

	GridSonelPage* pages;
	uint64_t size;
};

