//
// Created by brent on 27/01/2022.
//

#include "SonelMapper2.h"

extern "C" char embedded_mapper_code[];

SonelMapper2::SonelMapper2(const OptixSetup &optixSetup, const OptixScene &optixScene): SmOptixProgram<CudaSonelMapperParams, EmptyRecord, EmptyRecord, TriangleMeshSbtData>(embedded_mapper_code, optixSetup, optixScene, 1, 1, 1) {
	hitAhEnabled = true;
}

void SonelMapper2::initialize(SonelMapperConfig2 config) {
	sonelMap.setSoundSources(config.soundSources);
	sonelMap.duration = config.echogramDuration;
	sonelMap.soundSpeed = config.soundSpeed;
	sonelMap.timestep = 0.01f;

	sonelMapDevicePtr = sonelMap.cudaCreate();

	launchParams.localFrequencyIndex = 0;
	launchParams.sonelMapData = sonelMapDevicePtr;
	launchParams.traversable = optixScene.getGeoTraversable();

	init();
}

void SonelMapper2::execute() {
	sonelArrays.resize(static_cast<uint64_t>(sonelMap.duration / sonelMap.timestep) + 1);

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
			downloadSonelDataForFrequency(fIndex, sourceIndex);
			sonelMap.cudaDestroy(sonelMapDevicePtr, sourceIndex, fIndex);
			cudaSyncCheck("SonelMapper", "Failed to sync.");
		}
	}
}

const char *SonelMapper2::getLaunchParamsName() {
	return "params";
}

void SonelMapper2::configureRaygenProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.raygen.entryFunctionName = "__raygen__generateSonelMap";
}

void SonelMapper2::configureMissProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.miss.entryFunctionName = "__miss__sonelRadiance";
}

void SonelMapper2::configureHitProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.hitgroup.entryFunctionNameCH = "__closesthit__sonelRadiance";
	desc.hitgroup.entryFunctionNameAH = "__anyhit__sonelRadiance";
}

void SonelMapper2::createHitRecords(std::vector<SmRecord<TriangleMeshSbtData>> &hitRecords) {
	const Model* model = optixScene.getModel();
	unsigned int numObjects = static_cast<unsigned int>(model->meshes.size());

	for (unsigned int meshId = 0; meshId < numObjects; meshId++) {
		for (unsigned int programId = 0; programId < hitgroupProgramSize; programId++) {
			SmRecord<TriangleMeshSbtData> rec;

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


void SonelMapper2::downloadSonelDataForFrequency(uint32_t fIndex, uint32_t sourceIndex) {
	SoundFrequency& frequency = sonelMap.soundSources[sourceIndex].frequencies[fIndex];

	sonelMap.cudaDownload(sonelMapDevicePtr, sourceIndex, fIndex);
	Sonel* sonels = frequency.sonels;

	// Go over all rays
	uint64_t sonelAmount = 0;
	for (uint32_t i = 0; i < frequency.sonelAmount; i++) {
		// Go over each bounce in the ray
		for (uint32_t j = 0; j < frequency.sonelMaxDepth; j++) {
			Sonel& sonel = sonels[i * frequency.sonelMaxDepth + j];
			sonel.frequency = frequency.frequency;

			// The energies of a sonel is 0 the ray is absorbed and done.
			if (sonel.energy < 0.00001f) {
				break;
			}

			auto timeIndex = static_cast<uint64_t>(sonel.time / sonelMap.timestep);

			if (timeIndex < sonelArrays.size()) {
				sonelAmount++;
				sonelArrays[timeIndex].push_back(sonel);
			}
		}
	}

	printf("[SonelMapper] Sonels added %llu\n", sonelAmount);
}

const SonelMapData &SonelMapper2::getSonelMapData() const {
	return sonelMap;
}

std::vector<std::vector<Sonel>> *SonelMapper2::getSonelArrays() {
	return &sonelArrays;
}


