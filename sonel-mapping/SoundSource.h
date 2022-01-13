#pragma once
#include "gdt/math/vec.h"
#include "optix.h"
#include "Sonel.h"

class SoundFrequency {
public:
	SoundFrequency() : SoundFrequency(0, 0, 0) {

	}

	SoundFrequency(uint32_t frequency, uint32_t sonelAmount, uint16_t sonelMaxDepth) :
		frequency(frequency),
		decibels(nullptr), decibelSize(0),
		sonels(nullptr), sonelSize(sonelAmount* sonelMaxDepth),
		sonelAmount(sonelAmount), sonelMaxDepth(sonelMaxDepth) {

	}

	SoundFrequency(uint32_t frequency, uint32_t sonelAmount, uint16_t sonelMaxDepth, const std::vector<float>& decibels) :
		SoundFrequency(frequency, sonelAmount, sonelMaxDepth) {

	}

	void destroy() {
		destroyDecibels();
		destroySonels();
	}

	void destroyDecibels() {
		if (decibels != nullptr) {
			delete[] decibels;
			decibels = nullptr;
			decibelSize = 0;
		}
	}

	void destroySonels() {
		if (sonels != nullptr) {
			delete[] sonels;
			sonels = nullptr;
		}
	}

	__host__ void setDecibels(const std::vector<float>& inputDecibels) {
		destroyDecibels();

		decibelSize = inputDecibels.size();
		decibels = new float[decibelSize];
		memcpy(decibels, inputDecibels.data(), decibelSize * sizeof(float));
	}

	SoundFrequency* cudaCreate() {
		SoundFrequency* deviceFrequency;

		cudaMalloc(&deviceFrequency, sizeof(SoundFrequency));
		cudaCopy(deviceFrequency);

		return deviceFrequency;
	}

	void cudaCopy(SoundFrequency* deviceFrequency) {
		cudaMemcpy(deviceFrequency, this, sizeof(SoundFrequency), cudaMemcpyHostToDevice);
	}

	void cudaUpload(SoundFrequency* devicePointer) {
		cudaCopy(devicePointer);

		// Allocate array and copy data over
		float* deviceDecibels;
		cudaMalloc(&deviceDecibels, sizeof(float) * decibelSize);
		cudaMemcpy(deviceDecibels, decibels, sizeof(float) * decibelSize, cudaMemcpyHostToDevice);

		// Allocate sonels, no need to copy as we only want to download
		Sonel* deviceSonels;
		cudaMalloc(&deviceSonels, sizeof(Sonel) * sonelSize);

		// Copy the pointers over
		cudaMemcpy(&(devicePointer->decibels), &deviceDecibels, sizeof(float*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devicePointer->sonels), &deviceSonels, sizeof(Sonel*), cudaMemcpyHostToDevice);
	}

	void cudaDownload(SoundFrequency* devicePointer) {
		destroySonels();
		sonels = new Sonel[sonelSize];

		SoundFrequency deviceCopy(0, 0, 0);
		cudaMemcpy(&deviceCopy, devicePointer, sizeof(SoundFrequency), cudaMemcpyDeviceToHost);

		cudaMemcpy(sonels, deviceCopy.sonels, sonelSize * sizeof(Sonel), cudaMemcpyDeviceToHost);
	}

	void cudaDestroy(SoundFrequency* devicePointer) {
		SoundFrequency deviceCopy(0, 0, 0);

		cudaMemcpy(&deviceCopy, devicePointer, sizeof(SoundFrequency), cudaMemcpyDeviceToHost);
		
		cudaFree(deviceCopy.sonels);
		cudaFree(deviceCopy.decibels);
	}

public:
	uint32_t frequency;
	
	// An array of decibel values, each index is one timestep 
	float* decibels;
	// Max size of the array
	uint32_t decibelSize;

	// The array to store the simulation in for these settings
	Sonel* sonels;
	// The size of the array
	uint32_t sonelSize;

	// The amount of sonels to simulate this sound
	uint32_t sonelAmount;
	// The max depth to simulate
	uint16_t sonelMaxDepth;
};



class SoundSource {
public:
	SoundSource(): frequencySize(0), frequencies(nullptr) {

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

		frequencySize = data.size();
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

	uint16_t frequencySize;
	SoundFrequency* frequencies;
};

