#pragma once

// our helper library for window handling
#include "GlfCameraWindow.h"
#include <gl/GL.h>
#include "../SonelMapping/SonelManager.h"
#include <chrono>

struct Model;
struct Camera;
struct QuadLight;

enum MainWindowMode {
    FRAME_BY_FRAME,
    PLAY
};

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

    void drawCanvas();

    void key(int key, int mods) override;

    vec2i fbSize;
	GLuint fbTexture{ 0 };
	SonelManager sonelManager;
	std::vector<uint32_t> pixels;

protected:
    MainWindowMode mode = FRAME_BY_FRAME;

    float targetFrameRate = 1.0f;
    float timeSinceLastFrame = 0.0f;
    bool renderFrame = true;
};