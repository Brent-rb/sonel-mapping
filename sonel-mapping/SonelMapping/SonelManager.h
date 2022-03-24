#pragma once
#include "OptixSetup.h"
#include "OptixScene.h"
#include "SonelMapper.h"
#include "SonelMapVisualizer.h"
#include "../UI/Camera.h"
#include "SonelMapReceiver.h"

class SonelManager {
public:
	SonelManager(Model* model, SonelMapperConfig sonelMapperConfig);
	~SonelManager();

	void calculate();
	void render();
    void nextFrame();
    void previousFrame();

	/*! resize frame buffer to given resolution */
	void resize(const vec2i& newSize);

	/*! download the rendered color buffer */
	void downloadPixels(uint32_t h_pixels[]);

	void setCamera(const Camera& camera);

protected:
	OptixSetup optixSetup;
	OptixScene optixScene;

	SonelMapper sonelMapper;
	SonelMapVisualizer sonelVisualizer;
	SonelMapReceiver sonelMapReceiver;

	std::vector<SimpleSoundSource> simpleSoundSources;
};

