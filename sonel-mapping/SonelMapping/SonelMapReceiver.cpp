//
// Created by brent on 27/01/2022.
//

#include "SonelMapReceiver.h"
#include <iostream>
#include <fstream>
#include "../Cuda/CudaSonelReceiverHelper.h"

extern "C" char embedded_receiver_code[];

SonelMapReceiver::SonelMapReceiver(
	const OptixSetup& optixSetup,
	OptixScene& optixScene
): SmOptixProgram<SonelReceiverParams, EmptyRecord, EmptyRecord, SmSbtData>(embedded_receiver_code, optixSetup, optixScene, 1, 1, 1) {
	maxTraversableGraphDepth = 3;
	hitIsEnabled = true;
	hitAhEnabled = true;
}

void SonelMapReceiver::initialize(SonelMapReceiverConfig newConfig) {
	config = newConfig;

	launchParams.frequencySize = newConfig.frequencySize;
	launchParams.duration = newConfig.duration;
	launchParams.soundSpeed = newConfig.soundSpeed;
	launchParams.timestep = newConfig.timestep;
	launchParams.timestepSize = newConfig.timestepSize;
	launchParams.sonelRadius = 10.0f;
	launchParams.soundSourceRadius = 10.0f;
	launchParams.rayAmount = newConfig.rayAmount;

	init();
}

/*! set camera to render with */
void SonelMapReceiver::setCamera(const Camera& camera) {
	launchParams.camera.position = camera.from;
	launchParams.camera.direction = normalize(camera.at - camera.from);
}

void SonelMapReceiver::execute() {
	configureScene();
	createHitRecords();
	initEchogram();

	for (uint32_t timeIndex = 0; timeIndex < 1; timeIndex++) {
		launchParams.timeOffset = timeIndex * config.timestep;

		printf("Simulating\n");
		simulate();
		printf("Adding to echogram\n");
		addLaunchToEchogram();
	}

	writeEchogram();
}

void SonelMapReceiver::simulate() {
	bufferSize = getDataArraySize() * config.rayAmount;
	energyBuffer.alloc(sizeof(float) * bufferSize);
	launchParams.energies = reinterpret_cast<float*>(energyBuffer.getCuDevicePointer());

	launchOptix(config.rayAmount, 1, 1);
	cudaSyncCheck("SonelMapReceiver", "Failed sync");
}

void SonelMapReceiver::configureScene() {
	optixScene.setSonels(sonels, 5.0f);
	optixScene.build();
	launchParams.traversable = optixScene.getInstanceHandle();
}

void SonelMapReceiver::initEchogram() {
	echogram.resize(config.timestepSize);
	for (unsigned int timestep = 0; timestep < config.timestepSize; timestep++) {
		echogram[timestep].resize(MAX_FREQUENCIES, 0.0f);
	}

	highestTimestep = 0;
}

void SonelMapReceiver::addLaunchToEchogram() {
	// Storage for current launch
	std::vector<float> rtData;
	std::vector<std::vector<float>> tempEchogram;
	std::vector<std::vector<float>> hits;
	rtData.resize(bufferSize);
	energyBuffer.download(rtData.data(), bufferSize);

	hits.resize(config.timestepSize);
	tempEchogram.resize(config.timestepSize);
	for (unsigned int timestep = 0; timestep < config.timestepSize; timestep++) {
		hits[timestep].resize(MAX_FREQUENCIES, 0.0f);
		tempEchogram[timestep].resize(MAX_FREQUENCIES, 0.0f);
	}

	uint32_t rayStride = getDataArraySize();
	for (unsigned int rayIndex = 0; rayIndex < config.rayAmount; rayIndex++) {
		uint32_t rayStart = rayIndex * rayStride;

		for (unsigned int frequencyIndex = 0; frequencyIndex < MAX_FREQUENCIES; frequencyIndex++) {
			for (unsigned int sonelIndex = 0; sonelIndex < MAX_SONELS; sonelIndex++) {
				uint32_t sonelStart = rayStart + getSonelIndex(frequencyIndex, sonelIndex);
				float timestamp = rtData[sonelStart + DATA_TIME_OFFSET];
				float energy = rtData[sonelStart + DATA_ENERGY_OFFSET];
				auto timeIndex = static_cast<uint32_t>(round(timestamp / config.timestep));

				// When we encounter a 0 value we are probably done with this frequency
				// Timestamp COULD be 0 but unlikely
				if (energy < 0.000000000001f || timestamp < 0.00000000001f) {
					// printf("Skipped %f %f\n", energy, timestamp);
					break;
				}

				// Outside of range
				if (timeIndex < 0 || timeIndex >= echogram.size()) {
					printf("Skipped because outside %d\n", timeIndex);
					continue;
				}

				tempEchogram[timeIndex][frequencyIndex] += energy;
				hits[timeIndex][frequencyIndex] += 1.0f;
				highestTimestep = max(timeIndex, highestTimestep);
			}
		}
	}

	float sonelArea = (10.0f * 10.0f * 3.141592653589793238f) * 0.01f;
	float brdf = 1.0f / (2.0f * 3.141592653589793238f);
	for (unsigned int time = 0; time < min(config.timestepSize, highestTimestep); time++) {
		for (unsigned int frequency = 0; frequency < config.frequencySize; frequency++) {
			float hit = hits[time][frequency];

			if (hit > 1.0f) {
				echogram[time][frequency] += ((tempEchogram[time][frequency]) / sonelArea) * brdf;
			}
		}
	}
}

void SonelMapReceiver::writeEchogram() {
	for (unsigned int timestep = 0; timestep < min(static_cast<uint32_t>(echogram.size()), highestTimestep); timestep++) {
		std::cout << timestep * config.timestep;
		for (unsigned int frequency = 0; frequency < launchParams.frequencySize; frequency++) {
			float energy = echogram[timestep][frequency];
			float minEnergy = 10.0f / config.rayAmount;
			double decibel = 0;

			if (energy >= minEnergy)
				decibel = (log10(energy * config.rayAmount) * 10.0);

			std::cout << "; " << decibel;
		}

		if (timestep != echogram.size() - 1) {
			std::cout << "\n";
		}
	}
}

const char *SonelMapReceiver::getLaunchParamsName() {
	return "params";
}

void SonelMapReceiver::configureModuleCompileOptions(OptixModuleCompileOptions &compileOptions) {
	SmOptixProgram::configureModuleCompileOptions(compileOptions);
}

void SonelMapReceiver::configurePipelineCompileOptions(OptixPipelineCompileOptions &pipelineOptions) {
	SmOptixProgram::configurePipelineCompileOptions(pipelineOptions);

	pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;\
	pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
}

void SonelMapReceiver::configurePipelineLinkOptions(OptixPipelineLinkOptions &pipelineLinkOptions) {
	SmOptixProgram::configurePipelineLinkOptions(pipelineLinkOptions);
}

void SonelMapReceiver::configureRaygenProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.raygen.entryFunctionName = "__raygen__renderFrame";
}

void SonelMapReceiver::configureMissProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.miss.entryFunctionName = "__miss__radiance";
}

void SonelMapReceiver::configureHitProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
	desc.hitgroup.entryFunctionNameIS = "__intersection__radiance";
}

void SonelMapReceiver::addHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords) {
	addGeometryHitRecords(hitRecords);
	addSonelHitRecords(hitRecords);
	addSoundSourceHitRecords(hitRecords);
}

void SonelMapReceiver::addGeometryHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords) {
	const Model* model = optixScene.getModel();
	auto meshSize = static_cast<uint32_t>(model->meshes.size());

	for (uint32_t meshId = 0; meshId < meshSize; meshId++) {
		for (uint32_t programIndex = 0; programIndex < hitgroupProgramSize; programIndex++) {
			SmRecord<SmSbtData> rec;

			optixCheck(
					optixSbtRecordPackHeader(
							hitgroupPgs[programIndex],
							&rec
					),
					"SonelMapReceiver",
					"Failed to create SBT Record Header"
			);

			optixScene.fill(meshId, rec.data);
			rec.data.type = GEOMETRY;
			rec.data.sonel = nullptr;
			rec.data.soundSource = nullptr;
			hitRecords.push_back(rec);
		}
	}
}

void SonelMapReceiver::addSonelHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords) {
	for (uint32_t sonelId = 0; sonelId < optixScene.getSonelSize(); sonelId++) {
		for (uint32_t programIndex = 0; programIndex < hitgroupProgramSize; programIndex++) {
			SmRecord<SmSbtData> rec;

			optixCheck(
					optixSbtRecordPackHeader(
							hitgroupPgs[programIndex],
							&rec
					),
					"SonelMapReceiver",
					"Failed to create SBT Record Header"
			);

			rec.data.type = SONEL;
			rec.data.sonel = reinterpret_cast<Sonel*>(optixScene.getSonelDevicePointer(sonelId));
			rec.data.soundSource = nullptr;
			hitRecords.push_back(rec);
		}
	}
}

void SonelMapReceiver::addSoundSourceHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords) {
	for (uint32_t sourceId = 0; sourceId < optixScene.getSoundSourceSize(); sourceId++) {
		for (uint32_t programIndex = 0; programIndex < hitgroupProgramSize; programIndex++) {
			SmRecord<SmSbtData> rec;

			optixCheck(
					optixSbtRecordPackHeader(
							hitgroupPgs[programIndex],
							&rec
					),
					"SonelMapReceiver",
					"Failed to create SBT Record Header"
			);

			rec.data.type = SOUND_SOURCE;
			rec.data.soundSource = reinterpret_cast<SimpleSoundSource*>(optixScene.getSoundSourceDevicePointer(sourceId));
			rec.data.sonel = nullptr;
			hitRecords.push_back(rec);
		}
	}
}

void SonelMapReceiver::setSonels(std::vector<Sonel> *sonels) {
	this->sonels = sonels;
}