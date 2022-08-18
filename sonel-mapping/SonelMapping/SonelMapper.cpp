//
// Created by brent on 27/01/2022.
//

#include "SonelMapper.h"

extern "C" char embedded_mapper_code[];

SonelMapper::SonelMapper(const OptixSetup &optixSetup, OptixScene &optixScene): SmOptixProgram<CudaSonelMapperParams, EmptyRecord, EmptyRecord, SmSbtData>(embedded_mapper_code, optixSetup, optixScene, 1, 1, 1) {
	hitAhEnabled = true;
}

void SonelMapper::initialize(SonelMapperConfig config) {
	maxTime = config.echogramDuration;

	sonelMap.setSoundSources(config.soundSources);
	sonelMap.duration = config.echogramDuration;
	sonelMap.soundSpeed = config.soundSpeed;
	sonelMap.timestep = config.timestep;

	sonelMapDevicePtr = sonelMap.cudaCreate();

	launchParams.localFrequencyIndex = 0;
	launchParams.sonelMapData = sonelMapDevicePtr;
	launchParams.traversable = optixScene.getGeometryHandle();
	launchParams.frameIndex = 0;

	init();
}

void SonelMapper::execute() {
	for (uint32_t sourceIndex = 0; sourceIndex < sonelMap.soundSourceSize; sourceIndex++) {
		launchParams.soundSourceIndex = sourceIndex;
		SoundSource& soundSource = sonelMap.soundSources[sourceIndex];
		printf("[SonelMapper] Simulating sound source %d\n", sourceIndex);

		for (uint32_t fIndex = 0; fIndex < soundSource.frequencySize; fIndex++) {
			SoundFrequency& frequency = soundSource.frequencies[fIndex];
			printf("\tSimulating frequency (%d, %d)\n", fIndex, frequency.frequency);

			sonelMap.cudaUpload(sonelMapDevicePtr, sourceIndex, fIndex);
			launchParams.localFrequencyIndex = fIndex;
			launchParams.globalFrequencyIndex = sonelMap.getFrequencyIndex(frequency.frequency);
			launchOptix(frequency.sonelAmount, frequency.decibelSize, 1);
			cudaSyncCheck("SonelMapper", "Failed to sync.");

			downloadSonelDataForFrequency(fIndex, sourceIndex);
			sonelMap.cudaDestroy(sonelMapDevicePtr, sourceIndex, fIndex);
			launchParams.frameIndex += frequency.sonelAmount * frequency.sonelMaxDepth * frequency.decibelSize;
		}
	}
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
	SoundFrequency& frequency = sonelMap.soundSources[sourceIndex].frequencies[fIndex];

	sonelMap.cudaDownload(sonelMapDevicePtr, sourceIndex, fIndex);
	Sonel* sonels = frequency.sonels;

	uint64_t sonelAmount = 0;
	const uint64_t decibelStride = frequency.sonelAmount * frequency.sonelMaxDepth;
	for(uint32_t decibelIndex = 0; decibelIndex < frequency.decibelSize; decibelIndex++) {
		for(uint32_t sonelIndex = 0; sonelIndex < frequency.sonelAmount; sonelIndex++) {
			// Go over each bounce in the ray
			for(uint32_t depth = 0; depth < frequency.sonelMaxDepth; depth++) {
				uint64_t index = decibelIndex * decibelStride + sonelIndex * frequency.sonelMaxDepth + depth;
				Sonel& sonel = sonels[index];
				sonel.frequency = frequency.frequency;

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

	printf("\tSonels added %llu\n", sonelAmount);
}

const SonelMapData &SonelMapper::getSonelMapData() const {
	return sonelMap;
}

std::vector<Sonel> *SonelMapper::getSonelArray() {
	return &sonelArray;
}


