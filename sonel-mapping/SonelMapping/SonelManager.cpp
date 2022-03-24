#include "SonelManager.h"
#include <optix_function_table_definition.h>
#include "../Cuda/CudaRandom.h"

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
   ){
	optixScene.setModel(model);
	optixScene.build();

	sonelMapper.initialize(sonelMapperConfig);

	sonelMapReceiver.initialize({
		300,
		sonelMapperConfig.soundSpeed,
		sonelMapperConfig.echogramDuration,
		sonelMapperConfig.timestep,
		sonelMapper.getSonelMapData().frequencySize,
		static_cast<uint32_t>(round(sonelMapperConfig.echogramDuration / sonelMapperConfig.timestep)),
		&sonelMapperConfig.soundSources
	});

	sonelVisualizer.initialize();
}

SonelManager::~SonelManager() {
}

void SonelManager::calculate() {
	sonelMapper.execute();

	std::vector<Sonel>* sonels = sonelMapper.getSonelArray();
	sonelMapReceiver.setSonels(sonels);
	simpleSoundSources = SimpleSoundSource::from(sonelMapper.getSonelMapData());
    optixScene.setSonels(sonels, 0.15f);
	optixScene.setSoundSources(&simpleSoundSources, 0.15f);
    optixScene.build();

	sonelMapReceiver.execute();

    sonelVisualizer.setFrequencySize(sonelMapper.getSonelMapData().frequencySize);
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
