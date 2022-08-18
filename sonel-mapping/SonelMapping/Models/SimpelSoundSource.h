//
// Created by brent on 07/02/2022.
//

#ifndef SONEL_MAPPING_SIMPELSOUNDSOURCE_H
#define SONEL_MAPPING_SIMPELSOUNDSOURCE_H

#include "AabbItem.h"
#include "SoundSource.h"
#include "SonelMap.h"

class SimpleSoundSource: public AabbItem {
public:
	SimpleSoundSource() = default;
	SimpleSoundSource(
			gdt::vec3f position, gdt::vec3f direction,
			uint32_t frequency, uint16_t frequencyIndex,
			float timestamp, uint32_t timestampIndex,
			float decibel, float absorption
			): position(position), direction(direction),
			frequency(frequency), frequencyIndex(frequencyIndex),
			timestamp(timestamp), timestampIndex(timestampIndex),
			decibel(decibel), absorption(absorption) {

	}

	static std::vector<SimpleSoundSource> from(const SonelMapData& sonelMap) {
		std::vector<SimpleSoundSource> simpleSources;
		SimpleSoundSource simpleSource;

		SoundSource* soundSources = sonelMap.soundSources;
		for(uint32_t sourceIndex = 0; sourceIndex < sonelMap.soundSourceSize; sourceIndex++) {
			SoundSource& soundSource = soundSources[sourceIndex];
			simpleSource.position = soundSource.position;
			simpleSource.direction = soundSource.direction;

			for (uint32_t frequencyIndex = 0; frequencyIndex < soundSource.frequencySize; frequencyIndex++) {
				SoundFrequency& soundFrequency = soundSource.frequencies[frequencyIndex];
				simpleSource.frequency = soundFrequency.frequency;
				simpleSource.frequencyIndex = sonelMap.getFrequencyIndex(simpleSource.frequency);
				simpleSource.absorption = soundFrequency.absorption;

				for (uint32_t decibelIndex = 0; decibelIndex < soundFrequency.decibelSize; decibelIndex++) {
					float decibel = soundFrequency.decibels[decibelIndex];
					if (decibel < 0.00000000001f) {
						continue;
					}

					simpleSource.timestampIndex = decibelIndex;
					simpleSource.timestamp = static_cast<float>(decibelIndex) * sonelMap.timestep;
					simpleSource.decibel = decibel;
					simpleSource.radius = soundSource.radius;

					simpleSources.push_back(simpleSource);
				}
			}
		}

		std::cout << "Constructed " << simpleSources.size() << " simple sound sources." << std::endl;

		return simpleSources;
	}

	gdt::vec3f getPosition() const override {
		return position;
	}

	float getRadius() const override {
		return radius;
	}

public:
	gdt::vec3f position = gdt::vec3f();
	gdt::vec3f direction = gdt::vec3f();

	uint32_t frequency = 0;
	uint16_t frequencyIndex = 0;

	float timestamp = 0.0f;
	uint32_t timestampIndex = 0;

	float decibel = 0.0f;
	float absorption = 0.0f;
	float radius = 0.5f;
};

#endif //SONEL_MAPPING_SIMPELSOUNDSOURCE_H
