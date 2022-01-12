#include "Sonel.h"
#include <iostream>
#include <format>

SonelMap::SonelMap(Sonel sonels[], uint32_t sonelAmount, uint32_t depth, float echogramDuration, float soundSpeed, gdt::box3f bounds) {
	BoundingBox bb = BoundingBox(bounds.lower, bounds.upper);
	
	resolution = 0.1;
	timestep = 0.005;

	uint64_t maxTimesteps = (uint64_t)ceil(echogramDuration / timestep) + 1;
	sonelMap.resize(maxTimesteps);
	for (int i = 0; i < maxTimesteps; i++) {
		sonelMap[i].init(bb, 20);
	}


	
	int rayIndex = 0;
	int rayDepth = 0;
	int timeIndex;
	int x, y, z;
	gdt::vec3f sonelPosition;
	std::map< std::string, std::vector<Sonel>>* tempMap;
	char keyBuffer[512];

	do {
		Sonel& tempSonel = sonels[rayIndex * depth + rayDepth];

		if (tempSonel.energy < 0.0001f) {
			rayIndex++;
			rayDepth = 0;
		}
		else {
			rayDepth++;

			timeIndex = round((tempSonel.distance / soundSpeed) / timestep);

			if (timeIndex < sonelMap.size())
				sonelMap[timeIndex].insert(&tempSonel, tempSonel.position);
		}
	}
	while (rayIndex < sonelAmount);
}

SonelMap::~SonelMap() {

}

OctTree<Sonel>& SonelMap::getTimestep(int index) {
	return sonelMap[index];
}

int SonelMap::getTimestepAmount() {
	return sonelMap.size();
}