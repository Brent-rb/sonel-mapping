#pragma once

// our helper library for window handling
#include "GlfCameraWindow.h"
#include <gl/GL.h>
#include "../SonelMapping/SonelManager.h"

struct Model;
struct Camera;
struct QuadLight;

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