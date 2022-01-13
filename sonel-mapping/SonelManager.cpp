#include "SonelManager.h"
#include <optix_function_table_definition.h>

SonelManager::SonelManager(
	const Model* model, 
	SonelMapperConfig sonelMapperConfig
): optixSetup(), 
	optixScene(optixSetup.getOptixContext(), model),
	sonelMapper(
		optixSetup,
		optixScene,
		sonelMapperConfig
	),
	sonelVisualizer(
		optixSetup,
		optixScene
	) {

}

SonelManager::~SonelManager() {
}

void SonelManager::calculate() {
	sonelMapper.calculate();
	sonelVisualizer.setSonelMap(sonelMapper.getSonelMap());
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
