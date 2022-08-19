//
// Created by brent on 27/01/2022.
//

#include "SonelMapper.h"
#include <chrono>
using namespace std::chrono;

extern "C" char embedded_mapper_code[];

SonelMapper::SonelMapper(const OptixSetup &optixSetup, OptixScene &optixScene): SmOptixProgram<CudaSonelMapperParams, EmptyRecord, EmptyRecord, SmSbtData>(embedded_mapper_code, optixSetup, optixScene, 1, 1, 1) {
	hitAhEnabled = true;
}

void SonelMapper::initialize(SonelMapperConfig config) {
	maxTime = config.echogramDuration;

	simulationData.setSoundSources(config.soundSources);
	simulationData.duration = config.echogramDuration;
	simulationData.soundSpeed = config.soundSpeed;
	simulationData.timestep = config.timestep;

	sonelMapDevicePtr = simulationData.cudaCreate();

	launchParams.localFrequencyIndex = 0;
	launchParams.sonelMapData = sonelMapDevicePtr;
	launchParams.traversable = optixScene.getGeometryHandle();
	launchParams.frameIndex = 0;

	init();
}

void SonelMapper::execute() {
	auto executeStart = high_resolution_clock::now();

	for (uint32_t sourceIndex = 0; sourceIndex < simulationData.soundSourceSize; sourceIndex++) {
		launchParams.soundSourceIndex = sourceIndex;
		SoundSource& soundSource = simulationData.soundSources[sourceIndex];
		printf("[SonelMapper] Simulating sound source %d\n", sourceIndex);

		for (uint32_t fIndex = 0; fIndex < soundSource.frequencySize; fIndex++) {
			SoundFrequency& frequency = soundSource.frequencies[fIndex];
			printf("\tSimulating frequency (%d, %d)\n", fIndex, frequency.frequency);

			auto uploadStart = high_resolution_clock::now();
			simulationData.cudaUpload(sonelMapDevicePtr, sourceIndex, fIndex);
			auto uploadEnd = high_resolution_clock::now();
			auto uploadDelta = uploadEnd - uploadStart;
			auto durationMs = duration_cast<milliseconds>(uploadDelta);
			auto durationS  = duration_cast<seconds>(uploadDelta);
			printf("[Time] Uploading Simulation data took %dms %fs\n", durationMs, durationS);

			launchParams.localFrequencyIndex = fIndex;
			launchParams.globalFrequencyIndex = simulationData.getFrequencyIndex(frequency.frequency);

			auto launchStart = high_resolution_clock::now();
			launchOptix(frequency.sonelAmount, frequency.decibelSize, 1);
			cudaSyncCheck("SonelMapper", "Failed to sync.");
			auto launchEnd = high_resolution_clock::now();
			auto launchDelta = launchEnd - launchStart;
			auto launchMs = duration_cast<milliseconds>(launchDelta);
			auto launchS = duration_cast<seconds>(launchDelta);
			printf("[Time] Tracing frequency took %dms %fs\n", launchMs, launchS);

			auto downloadStart = high_resolution_clock::now();
			downloadSonelDataForFrequency(fIndex, sourceIndex);
			auto downloadEnd = high_resolution_clock::now();
			auto downloadDelta = downloadEnd - downloadStart;
			auto downloadMs = duration_cast<milliseconds>(downloadDelta);
			auto downloadS = duration_cast<seconds>(downloadDelta);
			printf("[Time] Downloading data for frequency took %dms %fs\n", downloadMs, downloadS);

			auto destroyStart = high_resolution_clock::now();
			simulationData.cudaDestroy(sonelMapDevicePtr, sourceIndex, fIndex);
			frequency.destroySonels();
			auto destroyEnd = high_resolution_clock::now();
			auto destroyDelta = destroyEnd - destroyStart;
			auto destroyMs = duration_cast<milliseconds>(destroyDelta);
			auto destroyS = duration_cast<seconds>(destroyDelta);
			printf("[Time] Destroying data for frequency took %dms %fs\n", destroyMs, destroyS);

			launchParams.frameIndex += frequency.sonelAmount * frequency.sonelMaxDepth * frequency.decibelSize;
		}
	}

	auto executeEnd = high_resolution_clock::now();
	auto executeDelta = executeEnd - executeStart;
	auto executeMs = duration_cast<milliseconds>(executeDelta);
	auto executeS = duration_cast<seconds>(executeDelta);
	printf("[Time] Tracing step took %dms %fs\n", executeMs, executeS);
}

const char *SonelMapper::getLaunchParamsName() {
	return "params";
}

void SonelMapper::configureRaygenProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.raygen.entryFunctionName = "__raygen__generateSonelMap";
}

void SonelMapper::configureMissProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.miss.entryFunctionName = "__miss__sonelRadiance";
}

void SonelMapper::configureHitProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.hitgroup.entryFunctionNameCH = "__closesthit__sonelRadiance";
	desc.hitgroup.entryFunctionNameAH = "__anyhit__sonelRadiance";
}

void SonelMapper::addHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords) {
	const Model* model = optixScene.getModel();
	auto numObjects = static_cast<unsigned int>(model->meshes.size());

	for (unsigned int meshId = 0; meshId < numObjects; meshId++) {
		for (unsigned int programId = 0; programId < hitgroupProgramSize; programId++) {
			SmRecord<SmSbtData> rec;

			optixCheck(
				optixSbtRecordPackHeader(hitgroupPgs[programId], &rec),
				"SonelMapper",
				"Failed to record sbt record header (hitgroup pgs)."
			);

			optixScene.fill(meshId, rec.data);

			hitRecords.push_back(rec);
		}
	}
}


void SonelMapper::downloadSonelDataForFrequency(uint32_t fIndex, uint32_t sourceIndex) {
	SoundFrequency& frequency = simulationData.soundSources[sourceIndex].frequencies[fIndex];

	auto destroyStart = high_resolution_clock::now();
	frequency.cudaDownloadSonels();
	auto destroyEnd = high_resolution_clock::now();
	auto destroyDelta = destroyEnd - destroyStart;
	auto destroyMs = duration_cast<milliseconds>(destroyDelta);
	auto destroyS = duration_cast<seconds>(destroyDelta);
	printf("	[Time] Downloading data for frequency took %dms %fs\n", destroyMs, destroyS);

	Sonel* sonels = frequency.sonels;

	uint64_t sonelAmount = 0;
	const uint64_t decibelStride = frequency.sonelAmount * frequency.sonelMaxDepth;
	for(uint32_t decibelIndex = 0; decibelIndex < frequency.decibelSize; decibelIndex++) {
		for(uint32_t sonelIndex = 0; sonelIndex < frequency.sonelAmount; sonelIndex++) {
			// Go over each bounce in the ray
			for(uint32_t depth = 0; depth < frequency.sonelMaxDepth; depth++) {
				uint64_t index = decibelIndex * decibelStride + sonelIndex * frequency.sonelMaxDepth + depth;
				Sonel& sonel = sonels[index];

				// The data of a sonel is 0 the ray is absorbed and done.
				if(sonel.frequency == 0) {
					break;
				}

				if(sonel.time < maxTime) {
					sonelAmount++;
					sonelArray.push_back(sonel);
				}
			}
		}
	}
}

SimulationData &SonelMapper::getSimulationData() {
	return simulationData;
}

std::vector<Sonel> *SonelMapper::getSonelArray() {
	return &sonelArray;
}


