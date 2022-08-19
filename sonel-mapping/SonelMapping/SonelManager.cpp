#include "SonelManager.h"
#include <optix_function_table_definition.h>
#include "../Cuda/CudaRandom.h"
#include <chrono>
using namespace std::chrono;

SonelManager::SonelManager(
		Model* model,
		SonelMapperConfig sonelMapperConfig
): optixSetup(), 
	optixScene(optixSetup.getOptixContext()),
	sonelMapper(
		optixSetup,
		optixScene
	),
	sonelVisualizer(
		optixSetup,
		optixScene,
        sonelMapperConfig.timestep,
        0.15f
	),
   sonelMapReceiver(
	   optixSetup,
	   optixScene
   ), absorptionData(50) {
	optixScene.setModel(model);
	optixScene.build();

	sonelMapper.initialize(sonelMapperConfig);

	SimulationData& simulationData = sonelMapper.getSimulationData();

	absorptionData.setAbsorptions(simulationData.frequencies, simulationData.frequencySize);

	sonelMapReceiver.initialize({
		3000,
		sonelMapperConfig.soundSpeed,
		sonelMapperConfig.echogramDuration,
		sonelMapperConfig.timestep,
		sonelMapper.getSimulationData().frequencySize,
		static_cast<uint32_t>(round(sonelMapperConfig.echogramDuration / sonelMapperConfig.timestep)),
		100,
		&simulationData,
		&absorptionData
	});

	sonelVisualizer.initialize();
}

SonelManager::~SonelManager() {
}

void SonelManager::calculate() {
	auto managerStart = high_resolution_clock::now();

	auto mapperStart = high_resolution_clock::now();
	sonelMapper.execute();
	auto mapperEnd = high_resolution_clock::now();
	auto mapperDelta = mapperEnd - mapperStart;
	auto mapperMs = duration_cast<milliseconds>(mapperDelta);
	printf("[Time] Mapper executing took %dms %fs\n", mapperMs, mapperMs / 1000.0f);


	auto prepStart = high_resolution_clock::now();
	std::vector<Sonel>* sonels = sonelMapper.getSonelArray();
	sonelMapReceiver.setSonels(sonels);
	simpleSoundSources = SimpleSoundSource::from(sonelMapper.getSimulationData());
	optixScene.setSonels(sonels, 0.15f);
	optixScene.setSoundSources(&simpleSoundSources);
    optixScene.build();
	auto prepEnd = high_resolution_clock::now();
	auto prepDelta = prepEnd - prepStart;
	auto prepMs = duration_cast<milliseconds>(prepDelta);
	printf("[Time] Preparing sonels and sound sources took %dms %fs\n", prepMs, prepMs / 1000.0f);

	auto receiverStart = high_resolution_clock::now();
	sonelMapReceiver.execute();
	auto receiverEnd = high_resolution_clock::now();
	auto receiverDelta = receiverEnd - receiverStart;
	auto receiverMs = duration_cast<milliseconds>(receiverDelta);
	printf("[Time] Receiver took %dms %fs\n", receiverMs, receiverMs / 1000.0f);

	auto managerEnd = high_resolution_clock::now();
	auto managerDelta = managerEnd - managerStart;
	auto managerMs = duration_cast<milliseconds>(managerDelta);
	printf("[Time] Sonel mapping took %dms %fs\n", managerMs, managerMs / 1000.0f);

    sonelVisualizer.setFrequencySize(sonelMapper.getSimulationData().frequencySize);
	sonelVisualizer.setSonelArray(sonels);
}

void SonelManager::render() {
	sonelVisualizer.execute();
}

void SonelManager::resize(const vec2i& newSize) {
	sonelVisualizer.resize(newSize);
}

void SonelManager::downloadPixels(uint32_t h_pixels[]) {
	sonelVisualizer.downloadPixels(h_pixels);
}

void SonelManager::setCamera(const Camera& camera) {
	sonelMapReceiver.setCamera(camera);
	sonelVisualizer.setCamera(camera);
}

void SonelManager::nextFrame() {
    sonelVisualizer.nextFrame();
}

void SonelManager::previousFrame() {
    sonelVisualizer.previousFrame();
}
