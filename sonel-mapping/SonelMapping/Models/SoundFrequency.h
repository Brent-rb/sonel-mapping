#pragma once
#include "gdt/math/vec.h"
#include "optix.h"
#include "Sonel.h"

class SoundFrequency {
public:
	SoundFrequency() : SoundFrequency(0, 0, 0) {

	}

	SoundFrequency(uint32_t frequency, uint32_t sonelAmount, uint16_t sonelMaxDepth) :
		frequency(frequency), sonels(nullptr), cudaSonels(nullptr),
		decibels(nullptr), decibelSize(0),
		sonelSize(sonelAmount* sonelMaxDepth),
		sonelAmount(sonelAmount), sonelMaxDepth(sonelMaxDepth) {}

	SoundFrequency(uint32_t frequency, uint32_t sonelAmount, uint16_t sonelMaxDepth, const std::vector<float>& decibels) :
		SoundFrequency(frequency, sonelAmount, sonelMaxDepth) {

	}

	void destroy() {
		destroyDecibels();
		destroySonels();
	}

	void destroyDecibels() {
		if(decibels != nullptr) {
			delete[] decibels;
			decibels = nullptr;
			decibelSize = 0;
		}
	}

	void destroySonels() {
		if(sonels != nullptr) {
			delete[] sonels;
			sonels = nullptr;
		}
	}

	__host__ void setDecibels(const std::vector<float>& inputDecibels) {
		destroyDecibels();

		decibelSize = static_cast<uint32_t>(inputDecibels.size());
		decibels = new float[decibelSize];
		memcpy(decibels, inputDecibels.data(), decibelSize * sizeof(float));

		sonelSize = decibelSize * sonelAmount * sonelMaxDepth;
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
		cudaSonels = deviceSonels;

		// Copy the pointers over
		cudaMemcpy(&(devicePointer->decibels), &deviceDecibels, sizeof(float*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devicePointer->sonels), &deviceSonels, sizeof(Sonel*), cudaMemcpyHostToDevice);
	}

	void cudaDownload(SoundFrequency* devicePointer) {
		destroySonels();
		sonels = new Sonel[sonelSize];

		SoundFrequency deviceCopy;
		cudaMemcpy(&deviceCopy, devicePointer, sizeof(SoundFrequency), cudaMemcpyDeviceToHost);

		cudaMemcpy(sonels, deviceCopy.sonels, sonelSize * sizeof(Sonel), cudaMemcpyDeviceToHost);
	}

	void cudaDownloadSonels() {
		sonels = new Sonel[sonelSize];
		cudaMemcpy(sonels, cudaSonels, sonelSize * sizeof(Sonel), cudaMemcpyDeviceToHost);
	}

	void cudaDestroy(SoundFrequency* devicePointer) {
		SoundFrequency deviceCopy;

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
	Sonel* cudaSonels;

	// The size of the array
	uint32_t sonelSize;

	// The amount of sonels to simulate this sound
	uint32_t sonelAmount;
	// The max depth to simulate
	uint16_t sonelMaxDepth;
};
