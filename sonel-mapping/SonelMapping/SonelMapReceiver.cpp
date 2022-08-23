//
// Created by brent on 27/01/2022.
//

#include "SonelMapReceiver.h"
#include <iostream>
#include <fstream>
#include "../Cuda/CudaSonelReceiverHelper.h"
#include <chrono>
#include <locale>
#include <format>
using namespace std::chrono;

extern "C" char embedded_receiver_code[];

#define SEARCH_RADIUS 0.15f

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
	launchParams.sonelRadius = SEARCH_RADIUS;
	launchParams.rayAmount = newConfig.rayAmount;
	launchParams.maxSonels = newConfig.maxSonels;
	launchParams.receiverRadius = 0.31f;
	launchParams.receiverPosition = gdt::vec3f(1.0, 1.0, -1.0);

	init();
}

/*! set camera to render with */
void SonelMapReceiver::setCamera(const Camera& camera) {
	launchParams.camera.position = camera.from;
	launchParams.camera.direction = normalize(camera.at - camera.from);
}

void SonelMapReceiver::execute(std::ofstream& timingFile) {
	auto configureStart = high_resolution_clock::now();
	configureScene();
	auto configureEnd = high_resolution_clock::now();
	auto configureDelta = configureEnd - configureStart;
	auto configureMs = duration_cast<microseconds>(configureDelta);
	timingFile << configureMs.count() / 1000.0f << "\t";
	printf("[Time][SonelReceiver][Configure] %f\n", configureMs.count() / 1000.0f);

	auto hitrecordStart = high_resolution_clock::now();
	createHitRecords();
	auto hitrecordEnd = high_resolution_clock::now();
	auto hitrecordDelta = hitrecordEnd - hitrecordStart;
	auto hitrecordMs = duration_cast<microseconds>(hitrecordDelta);
	timingFile << hitrecordMs.count() / 1000.0f << "\t";
	printf("[Time][SonelReceiver][HitRecords] %f\n", hitrecordMs.count() / 1000.0f);

	initEchogram();

	auto mapperStart = high_resolution_clock::now();
    simulate();
	auto mapperEnd = high_resolution_clock::now();
	auto mapperDelta = mapperEnd - mapperStart;
	auto mapperMs = duration_cast<microseconds>(mapperDelta);
	timingFile << mapperMs.count() / 1000.0f << "\t";
	printf("[Time][SonelReceiver][Launch] %f\n", mapperMs.count() / 1000.0f);

	auto echogramStart = high_resolution_clock::now();
    addLaunchToEchogram();
	auto echogramEnd = high_resolution_clock::now();
	auto echogramDelta = echogramEnd - echogramStart;
	auto echogramMs = duration_cast<microseconds>(echogramDelta);
	timingFile << echogramMs.count() / 1000.0f << "\t";
	printf("[Time][SonelReceiver][Echogram] %f\n", echogramMs.count() / 1000.0f);

	auto writeStart = high_resolution_clock::now();
	writeEchogram();
	auto writeEnd = high_resolution_clock::now();
	auto writeDelta = writeEnd - writeStart;
	auto writeMs = duration_cast<microseconds>(writeDelta);
	timingFile << writeMs.count() / 1000.0f << "\t";
	printf("[Time][SonelReceiver][I/O] %f\n", writeMs.count() / 1000.0f);
}

void SonelMapReceiver::simulate() {
	entriesBuffer.alloc(sizeof(GatherEntry) * config.frequencySize * config.rayAmount * config.maxSonels);
	hitBuffer.alloc(sizeof(uint16_t) * config.frequencySize * config.rayAmount);
	
	std::vector<float> absorptionVector(config.absorptionData->absorptions, config.absorptionData->absorptions + config.absorptionData->absorptionSize);
	absorptionBuffer.allocAndUpload(absorptionVector);

	launchParams.absorptionArray = reinterpret_cast<float*>(absorptionBuffer.cudaPointer);
	launchParams.entryBuffer = reinterpret_cast<GatherEntry*>(entriesBuffer.cudaPointer);
	launchParams.hitBuffer = reinterpret_cast<uint16_t*>(hitBuffer.cudaPointer);

	launchOptix(config.rayAmount, config.frequencySize, 1);
	cudaSyncCheck("SonelMapReceiver", "Failed sync");
}

void SonelMapReceiver::configureScene() {
	optixScene.build();
	launchParams.traversable = optixScene.getInstanceHandle();
}

void SonelMapReceiver::initEchogram() {
	echogram.resize(config.simulationData->frequencySize);
	for (unsigned int frequency = 0; frequency < config.simulationData->frequencySize; frequency++) {
		echogram[frequency].resize(config.timestepSize, 0.0f);
	}

	highestTimestep = 0;
}

void SonelMapReceiver::addLaunchToEchogram() {
	bool originalAlgorithm = true;

	// Storage for current launch
	std::vector<GatherEntry> gatherEntries;
	std::vector<uint16_t> hits;

	uint64_t entryBufferSize = config.frequencySize * config.rayAmount * config.maxSonels;
	uint64_t hitBufferSize = config.frequencySize * config.rayAmount;
	uint32_t maxTimesteps = config.timestepSize;

	gatherEntries.resize(entryBufferSize);
	hits.resize(hitBufferSize);

	entriesBuffer.download(gatherEntries.data(), entryBufferSize);
	hitBuffer.download(hits.data(), hitBufferSize);

	uint32_t directHits = 0;
	uint32_t maxHits = 0;

	for(uint32_t frequencyIndex = 0; frequencyIndex < config.simulationData->frequencySize; frequencyIndex++) {
		uint32_t frequency = config.simulationData->frequencies[frequencyIndex];

		for(uint32_t rayIndex = 0; rayIndex < config.rayAmount; rayIndex++) {
			uint64_t rayStart = frequencyIndex * config.rayAmount * config.maxSonels + rayIndex * config.maxSonels; 
			uint32_t rayHits = hits[frequencyIndex * config.rayAmount + rayIndex];
			float maxDistance = 0.0f;

			if(rayHits == config.maxSonels) {
				maxHits++;
			}

			if(!originalAlgorithm) {
				for(uint32_t hitIndex = 0; hitIndex < rayHits; hitIndex++) {
					GatherEntry& entry = gatherEntries[rayStart + hitIndex];
					maxDistance = max(maxDistance, entry.distance);
				}
			}


			float area = (M_PI * maxDistance * maxDistance);
			for(uint32_t hitIndex = 0; hitIndex < rayHits; hitIndex++) {
				GatherEntry& entry = gatherEntries[rayStart + hitIndex];

				auto timeIndex = static_cast<uint32_t>(floor((entry.time / config.timestep)));
				if(timeIndex < 0 || timeIndex >= config.timestepSize) {
					continue;
				}

				float scaling = static_cast<float>(rayHits);
				if(!originalAlgorithm) {
					scaling = area * config.rayAmount;
				}
				
				highestTimestep = max(timeIndex, highestTimestep);

				if(entry.distance < 1e-16) {
					directHits++;
					echogram[frequencyIndex][timeIndex] += entry.energy;
				}
				else {
					echogram[frequencyIndex][timeIndex] += entry.energy / scaling;
				}
			}
		}
	}

	printf("[SonelMapReceiver] Sound source hits %d, Saturated rays %f.\n", directHits, static_cast<float>(maxHits) / static_cast<float>(config.rayAmount) * 100.0f);
}

void SonelMapReceiver::writeEchogram() {
	auto sec_since_epoch = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();

	std::ofstream outputFile;
	std::stringstream outputName;
	outputName << "C:\\Users\\brent\\Desktop\\echograms\\echogram_" << sec_since_epoch << ".csv";

	outputFile.imbue(std::locale("fr"));
	outputFile.open(outputName.str());


	for(unsigned int frequency = 0; frequency < launchParams.frequencySize; frequency++) {
		for (unsigned int timestep = 0; timestep < min(config.timestepSize, highestTimestep); timestep++) {
			float energy = echogram[frequency][timestep];
			float minEnergy = 1e-3;
			double decibel = 0;

			if (energy >= minEnergy)
				decibel = (log10(energy / 1e-3) * 10.0);

			outputFile << decibel << "\t";
		}

		outputFile << "\n";
	}

	outputFile.close();
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

            // printf("[SonelMapReceiver] Added SoundSource %d\n", sourceId);
		}
	}
}

void SonelMapReceiver::setSonels(std::vector<Sonel> *sonels) {
	this->sonels = sonels;
}