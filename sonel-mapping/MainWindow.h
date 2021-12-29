#pragma once

// our helper library for window handling
#include "GlfCameraWindow.h"
#include <GL/gl.h>
#include "SonelMapper.h"

class Model;
class Camera;
class QuadLight;

class MainWindow: public GlfCameraWindow {
public:
	MainWindow(
		const std::string& title,
		const Model* model,
		const Camera& camera,
		const QuadLight& light,
		const float worldScale
	);

	virtual void render() override;
	virtual void draw() override;
	virtual void resize(const vec2i& newSize);

	vec2i fbSize;
	GLuint fbTexture{ 0 };
	SonelMapper mainWindow;
	std::vector<uint32_t> pixels;
};