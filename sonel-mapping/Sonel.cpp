#include "Sonel.h"
#include <iostream>
#include <format>

SonelMap::SonelMap(Sonel sonels[], uint32_t sonelAmount, uint32_t depth, float echogramDuration, float soundSpeed) {
	resolution = 0.1;
	timestep = 0.005;

	uint64_t maxTimesteps = (uint64_t)ceil(echogramDuration / timestep) + 1;
	sonelMap.resize(maxTimesteps);
	for (int i = 0; i < maxTimesteps; i++) {
		sonelMap[i] = std::vector<Sonel>();
	}


	Sonel* tempSonel;
	int rayIndex = 0;
	int rayDepth = 0;
	int timeIndex;
	int x, y, z;
	gdt::vec3f sonelPosition;
	std::map< std::string, std::vector<Sonel>>* tempMap;
	char keyBuffer[512];

	do {
		tempSonel = &sonels[rayIndex * depth + rayDepth];

		if (tempSonel->energy < 0.0001f) {
			rayIndex++;
			rayDepth = 0;
		}
		else {
			rayDepth++;

			timeIndex = round((tempSonel->distance / soundSpeed) / timestep);

			if (timeIndex < sonelMap.size())
				sonelMap[timeIndex].push_back(*tempSonel);
		}
	}
	while (rayIndex < sonelAmount);
}

SonelMap::~SonelMap() {

}

void SonelMap::getTimestep(int index, std::vector<Sonel>& sonels) {
	sonels = sonelMap[index];
}

int SonelMap::getTimestepAmount() {
	return sonelMap.size();
}