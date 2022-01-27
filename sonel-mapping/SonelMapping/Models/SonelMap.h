#pragma once

#include <set>
#include "Sonel.h"
#include "SoundSource.h"

class SonelMapData {
public:
	SonelMapData(): 
		timestep(0.01f), duration(6.0f), soundSpeed(343.0f), 
		soundSources(nullptr), soundSourceSize(0) ,
        frequencies(nullptr), frequencySize(0) {

	}

	void destroy() {
		if (soundSources != nullptr) {
			delete[] soundSources;
			soundSources = nullptr;
			soundSourceSize = 0;
		}

        if (frequencies != nullptr) {
            delete[] frequencies;
            frequencies = nullptr;
            frequencySize = 0;
        }
	}
	
	__host__ void setSoundSources(const std::vector<SoundSource>& inputSources) {
		destroy();

        determineFrequencies(inputSources);

		soundSourceSize = static_cast<uint32_t>(inputSources.size());
		soundSources = new SoundSource[soundSourceSize];
		memcpy(soundSources, inputSources.data(), soundSourceSize * sizeof(SoundSource));
	}

	SonelMapData* cudaCreate() {
		SonelMapData* deviceSonelMap;

		cudaMalloc(&deviceSonelMap, sizeof(SonelMapData));
		cudaCopy(deviceSonelMap);

		return deviceSonelMap;
	}

	void cudaCopy(SonelMapData* deviceSonelMap) {
		cudaMemcpy(deviceSonelMap, this, sizeof(SonelMapData), cudaMemcpyHostToDevice);
	}

	void cudaUpload(SonelMapData* deviceSonelMap, uint16_t sourceIndex, uint16_t frequencyIndex) {
		cudaCopy(deviceSonelMap);

		SoundSource* deviceSoundSources;
		cudaMalloc(&deviceSoundSources, soundSourceSize * sizeof(SoundSource));
		cudaMemcpy(deviceSoundSources, this->soundSources, soundSourceSize * sizeof(SoundSource), cudaMemcpyHostToDevice);
		soundSources[sourceIndex].cudaUpload(&(deviceSoundSources[sourceIndex]), frequencyIndex);
		cudaMemcpy(&(deviceSonelMap->soundSources), &deviceSoundSources, sizeof(SoundSource*), cudaMemcpyHostToDevice);

        uint32_t* deviceFrequencies;
        cudaMalloc(&deviceFrequencies, frequencySize * sizeof(uint32_t));
        cudaMemcpy(deviceFrequencies, this->frequencies, frequencySize * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(&(deviceSonelMap->frequencies), &deviceFrequencies, sizeof(uint32_t*), cudaMemcpyHostToDevice);
	}

	void cudaDownload(SonelMapData* deviceSonelMap, uint16_t sourceIndex, uint16_t frequencyIndex) {
		SonelMapData deviceCopy;
		cudaMemcpy(&deviceCopy, deviceSonelMap, sizeof(SonelMapData), cudaMemcpyDeviceToHost);

		soundSources[sourceIndex].cudaDownload(&(deviceCopy.soundSources[sourceIndex]), frequencyIndex);
	}

	void cudaDestroy(SonelMapData* deviceSonelMap, uint16_t sourceIndex, uint16_t frequencyIndex) {
		SonelMapData deviceCopy;
		cudaMemcpy(&deviceCopy, deviceSonelMap, sizeof(SonelMapData), cudaMemcpyDeviceToHost);

		soundSources[sourceIndex].cudaDestroy(&(deviceCopy.soundSources[sourceIndex]), frequencyIndex);

		cudaFree(deviceCopy.soundSources);
        cudaFree(deviceCopy.frequencies);
	}

    uint16_t getFrequencyIndex(uint32_t frequency) const {
        for (int i = 0; i < frequencySize; i++) {
            if (frequency == frequencies[i]) {
                return i;
            }
        }

        return UINT16_MAX;
    }

private:
    __host__ void determineFrequencies(const std::vector<SoundSource>& inputSources) {
        std::set<uint32_t> frequencySet;
        std::vector<uint32_t> tempFrequencies;

        for (int i = 0; i < inputSources.size(); i++) {
            const SoundSource& source = inputSources[i];

            for (int f = 0; f < source.frequencySize; f++) {
                const SoundFrequency& frequency = source.frequencies[f];

                if (frequencySet.count(frequency.frequency) == 0) {
                    tempFrequencies.push_back(frequency.frequency);
                    frequencySet.insert(frequency.frequency);
                }
            }
        }

        std::sort(tempFrequencies.begin(), tempFrequencies.end());
        frequencies = new uint32_t[tempFrequencies.size()];
        memcpy(frequencies, tempFrequencies.data(), sizeof(uint32_t) * tempFrequencies.size());
        frequencySize = static_cast<uint32_t>(tempFrequencies.size());
    }


public:
	float timestep;
	float duration;
	float soundSpeed;

	SoundSource* soundSources;
	uint32_t soundSourceSize;
    uint32_t* frequencies;
    uint16_t frequencySize;
};
