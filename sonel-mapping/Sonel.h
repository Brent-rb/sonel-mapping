#pragma once
#include "gdt/math/vec.h"
#include <map>
#include <vector>

struct Sonel {
	gdt::vec3f position;
	gdt::vec3f incidence;
	float energy;
	float distance;
	uint32_t frequency;
};

class SonelMap {
public:
	SonelMap(Sonel* sonels, uint32_t sonelAmount, uint32_t maxDepth, float echogramDuration, float soundSpeed);
	~SonelMap();

	void getTimestep(int index, std::vector<Sonel>& sonels);
	int getTimestepAmount();

protected:
	// The sonels are first grouped by timestep (in discrete times)
	// Then they are grouped by resolution.
	// The coordinates will be changed from floats to discrete buckets according to the resolution.
	std::vector<std::vector<Sonel>> sonelMap;

	float resolution;
	float timestep;
};
