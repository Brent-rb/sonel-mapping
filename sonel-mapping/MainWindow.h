#pragma once

// our helper library for window handling
#include "GlfCameraWindow.h"
#include <GL/gl.h>
#include "SonelManager.h"

class Model;
class Camera;
class QuadLight;

class MainWindow: public GlfCameraWindow {
public:
	MainWindow(
		const std::string& title,
		Model* model,
		const Camera& camera,
		const float worldScale,
		SonelMapperConfig config
	);

	virtual void render() override;
	virtual void draw() override;
	virtual void resize(const vec2i& newSize);

	vec2i fbSize;
	GLuint fbTexture{ 0 };
	SonelManager sonelManager;
	std::vector<uint32_t> pixels;
};