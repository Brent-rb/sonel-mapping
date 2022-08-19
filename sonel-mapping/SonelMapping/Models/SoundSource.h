#pragma once
#include "gdt/math/vec.h"
#include "optix.h"
#include "Sonel.h"
#include "SoundFrequency.h"

class SoundSource {
public:
	SoundSource(): frequencySize(0), frequencies(nullptr), radius(0.0f) {

	}

	void destroy() {
		if (frequencies != nullptr) {
			delete[] frequencies;
			frequencySize = 0;
			frequencies = nullptr;
		}
	}

	void setFrequencies(const std::vector<SoundFrequency>& data) {
		destroy();

		frequencySize = static_cast<uint32_t>(data.size());
		frequencies = new SoundFrequency[frequencySize];
		memcpy(this->frequencies, data.data(), frequencySize * sizeof(SoundFrequency));
	}

	SoundSource* cudaCreate() {
		SoundSource* deviceSoundSource;

		cudaMalloc(&deviceSoundSource, sizeof(SoundSource));
		cudaCopy(deviceSoundSource);

		return deviceSoundSource;
	}

	void cudaCopy(SoundSource* deviceSoundSource) {
		cudaMemcpy(deviceSoundSource, this, sizeof(SoundSource), cudaMemcpyHostToDevice);
	}

	void cudaUpload(SoundSource* devicePointer, uint16_t frequencyIndex) {
		cudaCopy(devicePointer);

		SoundFrequency* deviceFrequencies;

		cudaMalloc(&deviceFrequencies, frequencySize * sizeof(SoundFrequency));
		cudaMemcpy(deviceFrequencies, frequencies, frequencySize * sizeof(SoundFrequency), cudaMemcpyHostToDevice);

		if (frequencyIndex < frequencySize) {
			frequencies[frequencyIndex].cudaUpload(&(deviceFrequencies[frequencyIndex]));
		}

		cudaMemcpy(&(devicePointer->frequencies), &deviceFrequencies, sizeof(SoundFrequency*), cudaMemcpyHostToDevice);
	}

	void cudaDownload(SoundSource* devicePointer, uint16_t frequencyIndex) {
		SoundSource deviceCopy;

		cudaMemcpy(&deviceCopy, devicePointer, sizeof(SoundSource), cudaMemcpyDeviceToHost);

		if (frequencyIndex < frequencySize) {
			frequencies[frequencyIndex].cudaDownload(&(deviceCopy.frequencies[frequencyIndex]));
		}
	}

	void cudaDownloadSonels(uint16_t frequencyIndex) {
		frequencies[frequencyIndex].cudaDownloadSonels();
	}

	void cudaDestroy(SoundSource* devicePointer, uint16_t frequencyIndex) {
		SoundSource deviceCopy;

		cudaMemcpy(&deviceCopy, devicePointer, sizeof(SoundSource), cudaMemcpyDeviceToHost);

		if (frequencyIndex < frequencySize) {
			frequencies[frequencyIndex].cudaDestroy(&(deviceCopy.frequencies[frequencyIndex]));
		}

		cudaFree(deviceCopy.frequencies);
	}

public:
	gdt::vec3f position;
	gdt::vec3f direction;
	float radius;

	uint16_t frequencySize;
	SoundFrequency* frequencies;
};

