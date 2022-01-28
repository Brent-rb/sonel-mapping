#include "SonelManager.h"
#include <optix_function_table_definition.h>

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
		optixScene
	),
   sonelMapReceiver(
	   optixSetup,
	   optixScene
   ){

	optixScene.setModel(model);
	optixScene.build();

	sonelMapper.initialize(sonelMapperConfig);
	sonelMapReceiver.initialize({
		100000,
		sonelMapperConfig.soundSpeed,
		sonelMapperConfig.echogramDuration,
		sonelMapperConfig.timestep,
		sonelMapper.getSonelMapData().frequencySize,
		static_cast<uint32_t>(round(sonelMapperConfig.echogramDuration / sonelMapperConfig.timestep))
	});
	// sonelVisualizer.initialize();
}

SonelManager::~SonelManager() {
}

void SonelManager::calculate() {
	sonelMapper.execute();

	std::vector<Sonel>* sonels = sonelMapper.getSonelArray();
	sonelMapReceiver.setSonels(sonels);
	sonelMapReceiver.execute();

    // sonelVisualizer.setFrequencySize(sonelMapper.getSonelMapData().frequencySize);
	// sonelVisualizer.setSonelArray(sonelMapper.getSonelArrays());
}

void SonelManager::render() {
	// sonelVisualizer.execute();
}

void SonelManager::resize(const vec2i& newSize) {
	// sonelVisualizer.resize(newSize);
}

void SonelManager::downloadPixels(uint32_t h_pixels[]) {
	// sonelVisualizer.downloadPixels(h_pixels);
}

void SonelManager::setCamera(const Camera& camera) {
	sonelMapReceiver.setCamera(camera);
	// sonelVisualizer.setCamera(camera);
}
