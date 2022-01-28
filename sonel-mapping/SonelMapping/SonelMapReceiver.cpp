//
// Created by brent on 27/01/2022.
//

#include "SonelMapReceiver.h"
#include <iostream>
#include <fstream>

extern "C" char embedded_receiver_code[];

#define MAX_FREQUENCIES 8
#define MAX_DEPTH 8

SonelMapReceiver::SonelMapReceiver(
	const OptixSetup& optixSetup,
	OptixScene& optixScene
): SmOptixProgram<SonelReceiverParams, EmptyRecord, EmptyRecord, TriangleMeshSbtData>(embedded_receiver_code, optixSetup, optixScene, 1, 1, 1) {
	maxTraversableGraphDepth = 3;
	hitIsEnabled = true;
	hitAhEnabled = true;
}

void SonelMapReceiver::initialize(SonelMapReceiverConfig config) {
	this->config = config;

	launchParams.frequencySize = config.frequencySize;
	launchParams.duration = config.duration;
	launchParams.soundSpeed = config.soundSpeed;
	launchParams.timestep = config.timestep;
	launchParams.timestepSize = config.timestepSize;

	init();
}

/*! set camera to render with */
void SonelMapReceiver::setCamera(const Camera& camera) {
	launchParams.camera.position = camera.from;
	launchParams.camera.direction = normalize(camera.at - camera.from);
}

void SonelMapReceiver::execute() {
	optixScene.setSonels(sonels, 5.0f);
	optixScene.build();
	launchParams.traversable = optixScene.getInstanceTraversable();

	createHitRecords();

	bufferSize = (MAX_FREQUENCIES + 1) * MAX_DEPTH * config.rayAmount;
	energyBuffer.alloc(sizeof(float) * bufferSize);
	launchParams.energies = reinterpret_cast<float*>(energyBuffer.getCuDevicePointer());

	launchOptix(config.rayAmount, 1, 1);
	cudaSyncCheck("SonelMapReceiver", "Failed sync");

	std::vector<float> rtData;
	std::vector<std::vector<float>> echogram;

	rtData.resize(bufferSize);
	energyBuffer.download(rtData.data(), bufferSize);

	echogram.resize(config.timestepSize);
	for (unsigned int timestep = 0; timestep < config.timestepSize; timestep++) {
		echogram[timestep].resize(MAX_FREQUENCIES, 0.0f);
	}

	uint32_t bounceStride = (MAX_FREQUENCIES + 1);
	uint32_t stride = bounceStride * MAX_DEPTH;
	for (unsigned int ray = 0; ray < config.rayAmount; ray++) {
		uint32_t rayStart = ray * stride;

		for (unsigned int bounce = 0; bounce < MAX_DEPTH; bounce++) {
			uint32_t bounceStart = bounce * bounceStride;
			float timestamp = rtData[rayStart + bounceStart];
			uint32_t timeIndex = static_cast<uint32_t>(round(timestamp / config.timestep));

			if (timeIndex >= echogram.size()) {
				continue;
			}

			for (unsigned int frequency = 0; frequency < MAX_FREQUENCIES; frequency++) {
				float energy = rtData[rayStart + bounceStart + frequency + 1];

				echogram[timeIndex][frequency] += energy;
			}
		}
	}

	// std::cout << "{[\n";
	for (unsigned int timestep = 0; timestep < echogram.size(); timestep++) {
		// std::cout << "\t[";
		for (unsigned int frequency = 0; frequency < launchParams.frequencySize; frequency++) {
			std::cout << echogram[timestep][frequency];

			if (frequency != launchParams.frequencySize - 1) {
				std::cout << ", ";
			}
		}
		//std::cout << "]";
		if (timestep != echogram.size() - 1) {
			std::cout << "\n";
		}
	}
	//std::cout << "]}\n";

	printf("Wrote out echogram.\n");
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

void SonelMapReceiver::addHitRecords(std::vector<SmRecord<TriangleMeshSbtData>> &hitRecords) {
	const Model* model = optixScene.getModel();
	auto meshSize = static_cast<uint32_t>(model->meshes.size());

	for (uint32_t meshId = 0; meshId < meshSize; meshId++) {
		for (uint32_t programIndex = 0; programIndex < hitgroupProgramSize; programIndex++) {
			SmRecord<TriangleMeshSbtData> rec;

			optixCheck(
				optixSbtRecordPackHeader(
					hitgroupPgs[programIndex],
					&rec
				),
				"SonelMapVisualizer",
				"Failed to create SBT Record Header"
			);

			optixScene.fill(meshId, rec.data);
			rec.data.sonel = nullptr;
			hitRecords.push_back(rec);
		}
	}

	for (uint32_t sonelId = 0; sonelId < optixScene.getSonelSize(); sonelId++) {
		for (uint32_t programIndex = 0; programIndex < hitgroupProgramSize; programIndex++) {
			SmRecord<TriangleMeshSbtData> rec;

			optixCheck(
			optixSbtRecordPackHeader(
					hitgroupPgs[programIndex],
					&rec
				),
				"SonelMapVisualizer",
				"Failed to create SBT Record Header"
			);

			rec.data.sonel = (Sonel*) optixScene.getSonelDevicePointer(sonelId);
			hitRecords.push_back(rec);
		}
	}
}

void SonelMapReceiver::setSonels(std::vector<Sonel> *sonels) {
	this->sonels = sonels;
}