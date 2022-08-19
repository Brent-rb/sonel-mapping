#pragma once

#include <set>
#include "Sonel.h"
#include "SoundSource.h"

class SimulationData {
public:
	SimulationData() :
		timestep(0.01f), duration(6.0f), soundSpeed(343.0f), humidity(50),
		soundSources(nullptr), soundSourceSize(0),
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
		checkFrequenciesAllowed(inputSources);

		soundSourceSize = static_cast<uint32_t>(inputSources.size());
		soundSources = new SoundSource[soundSourceSize];
		memcpy(soundSources, inputSources.data(), soundSourceSize * sizeof(SoundSource));
	}

	SimulationData* cudaCreate() {
		SimulationData* deviceSonelMap;

		cudaMalloc(&deviceSonelMap, sizeof(SimulationData));
		cudaCopy(deviceSonelMap);

		return deviceSonelMap;
	}

	void cudaCopy(SimulationData* deviceSonelMap) {
		cudaMemcpy(deviceSonelMap, this, sizeof(SimulationData), cudaMemcpyHostToDevice);
	}

	void cudaUpload(SimulationData* deviceSonelMap, uint16_t sourceIndex, uint16_t frequencyIndex) {
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

	void cudaDownload(SimulationData* deviceSonelMap, uint16_t sourceIndex, uint16_t frequencyIndex) {
		SimulationData deviceCopy;
		cudaMemcpy(&deviceCopy, deviceSonelMap, sizeof(SimulationData), cudaMemcpyDeviceToHost);

		soundSources[sourceIndex].cudaDownload(&(deviceCopy.soundSources[sourceIndex]), frequencyIndex);
	}

	void cudaDownloadSonels(uint16_t sourceIndex, uint16_t frequencyIndex) {
		soundSources[sourceIndex].cudaDownloadSonels(frequencyIndex);
	}

	void cudaDestroy(SimulationData* deviceSonelMap, uint16_t sourceIndex, uint16_t frequencyIndex) {
		SimulationData deviceCopy;
		cudaMemcpy(&deviceCopy, deviceSonelMap, sizeof(SimulationData), cudaMemcpyDeviceToHost);

		soundSources[sourceIndex].cudaDestroy(&(deviceCopy.soundSources[sourceIndex]), frequencyIndex);

		cudaFree(deviceCopy.soundSources);
        cudaFree(deviceCopy.frequencies);
		deviceCopy.soundSources = nullptr;
		deviceCopy.frequencies = nullptr;
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
		frequencySize = static_cast<uint16_t>(tempFrequencies.size());

		
    }

	__host__ void checkFrequenciesAllowed(const std::vector<SoundSource>& inputSources) {
		std::set<uint16_t> allowedFrequencySet(allowedFrequencies, allowedFrequencies + 7);

		for(int i = 0; i < inputSources.size(); i++) {
			const SoundSource& source = inputSources[i];

			for(int f = 0; f < source.frequencySize; f++) {
				const SoundFrequency& frequency = source.frequencies[f];

				if(allowedFrequencySet.count(frequency.frequency) == 0) {
					std::cerr << "Frequency " << frequency.frequency << " is not allowed." << std::endl;
					exit(EXIT_FAILURE);
				}
			}
		}
	}

public:
	float timestep;
	float duration;
	float soundSpeed;
	uint16_t humidity;

	SoundSource* soundSources;
	uint32_t soundSourceSize;
    uint32_t* frequencies;
    uint16_t frequencySize;

	const uint16_t allowedFrequencies[7] = {63, 125, 250, 500, 1000, 2000, 4000};
};
