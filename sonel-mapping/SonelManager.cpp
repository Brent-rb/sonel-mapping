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
	) {

	optixScene.setModel(model);
	optixScene.build();

	sonelMapper.init(sonelMapperConfig);
	sonelVisualizer.init();
}

SonelManager::~SonelManager() {
}

void SonelManager::calculate() {
	sonelMapper.calculate();
	sonelVisualizer.setSonelMap(sonelMapper.getSonelMap());
	sonelVisualizer.setSonelArray(sonelMapper.getSonelArrays());
}

void SonelManager::render() {
	sonelVisualizer.render();
}

void SonelManager::resize(const vec2i& newSize) {
	sonelVisualizer.resize(newSize);
}

void SonelManager::downloadPixels(uint32_t h_pixels[]) {
	sonelVisualizer.downloadPixels(h_pixels);
}

void SonelManager::setCamera(const Camera& camera) {
	sonelVisualizer.setCamera(camera);
}
