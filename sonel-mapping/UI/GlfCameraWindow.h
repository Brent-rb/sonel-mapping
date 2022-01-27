#pragma once

#include "gdt/math/vec.h"
#include "GlfWindow.h"

struct GlfCameraWindow : public GLFWindow {
	GlfCameraWindow(
		const std::string& title,
		const gdt::vec3f& camera_from,
		const gdt::vec3f& camera_at,
		const gdt::vec3f& camera_up,
		const float worldScale
	);

	void enableFlyMode();
	void enableInspectMode();

	// /*! put pixels on the screen ... */
	// virtual void draw()
	// { /* empty - to be subclassed by user */ }

	// /*! callback that window got resized */
	// virtual void resize(const vec2i &newSize)
	// { /* empty - to be subclassed by user */ }

	virtual void key(int key, int mods) override;

	/*! callback that window got resized */
	virtual void mouseMotion(const vec2i& newPos) override;

	/*! callback that window got resized */
	virtual void mouseButton(int button, int action, int mods) override;

	// /*! mouse got dragged with left button pressedn, by 'delta'
	//   pixels, at last position where */
	// virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta) {}

	// /*! mouse got dragged with left button pressedn, by 'delta'
	//   pixels, at last position where */
	// virtual void mouseDragRight (const vec2i &where, const vec2i &delta) {}

	// /*! mouse got dragged with left button pressedn, by 'delta'
	//   pixels, at last position where */
	// virtual void mouseDragMiddle(const vec2i &where, const vec2i &delta) {}


	/*! a (global) pointer to the currently active window, so we can
		route glfw callbacks to the right GLFWindow instance (in this
		simplified library we only allow on window at any time) */
		// static GLFWindow *current;

	struct {
		bool leftButton{ false }, middleButton{ false }, rightButton{ false };
	} isPressed;
	vec2i lastMousePos = { -1,-1 };

	friend struct CameraFrameManip;

	CameraFrame cameraFrame;
	std::shared_ptr<CameraFrameManip> cameraFrameManip;
	std::shared_ptr<CameraFrameManip> inspectModeManip;
	std::shared_ptr<CameraFrameManip> flyModeManip;
};