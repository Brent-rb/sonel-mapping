#pragma once

#include "Sonel.h"
#include "SoundSource.h"

class SonelMapData {
public:
	SonelMapData(): 
		timestep(0.01f), duration(6.0f), soundSpeed(343.0f), 
		soundSources(nullptr), soundSourceSize(0) {

	}

	void destroy() {
		if (soundSources != nullptr) {
			delete[] soundSources;
			soundSources = nullptr;
			soundSourceSize = 0;
		}
	}
	
	__host__ void setSoundSources(const std::vector<SoundSource>& inputSources) {
		destroy();

		soundSourceSize = inputSources.size();
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
	}

	void cudaDownload(SonelMapData* deviceSonelMap, uint16_t sourceIndex, uint16_t frequencyIndex) {
		SonelMapData deviceCopy;
		cudaMemcpy(&deviceCopy, deviceSonelMap, sizeof(SonelMapData), cudaMemcpyDeviceToHost);

		soundSources[sourceIndex].cudaDownload(&(deviceCopy.soundSources[sourceIndex]), frequencyIndex);
	}

	void cudaDestroy(SonelMapData* deviceSonelMap, uint16_t frequencyIndex) {
		SonelMapData deviceCopy;
		cudaMemcpy(&deviceCopy, deviceSonelMap, sizeof(SonelMapData), cudaMemcpyDeviceToHost);

		for (int i = 0; i < soundSourceSize; i++) {
			soundSources[i].cudaDestroy(&(deviceCopy.soundSources[i]), frequencyIndex);
		}

		cudaFree(deviceCopy.soundSources);
	}

public:
	float timestep;
	float duration;
	float soundSpeed;

	SoundSource* soundSources;
	uint32_t soundSourceSize;
};
