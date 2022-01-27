#pragma once
#include "gdt/math/vec.h"

struct CameraFrame;

// ------------------------------------------------------------------
/*! abstract base class that allows to manipulate a renderable
	camera */
struct CameraFrameManip {
	CameraFrameManip(CameraFrame* cameraFrame);

	/*! this gets called when the user presses a key on the keyboard ... */
	virtual void key(int key, int mods);
	virtual void strafe(const gdt::vec3f& howMuch);
	/*! strafe, in screen space */
	virtual void strafe(const gdt::vec2f& howMuch);

	virtual void move(const float step) = 0;
	virtual void rotate(const float dx, const float dy) = 0;

	// /*! this gets called when the user presses a key on the keyboard ... */
	// virtual void special(int key, const vec2i &where) { };

	/*! mouse got dragged with left button pressedn, by 'delta'
		pixels, at last position where */
	virtual void mouseDragLeft(const gdt::vec2f& delta);

	/*! mouse got dragged with left button pressedn, by 'delta'
		pixels, at last position where */
	virtual void mouseDragMiddle(const gdt::vec2f& delta); 

	/*! mouse got dragged with left button pressedn, by 'delta'
		pixels, at last position where */
	virtual void mouseDragRight(const gdt::vec2f& delta);

	// /*! mouse button got either pressed or released at given location */
	// virtual void mouseButtonLeft  (const vec2i &where, bool pressed) {}

	// /*! mouse button got either pressed or released at given location */
	// virtual void mouseButtonMiddle(const vec2i &where, bool pressed) {}

	// /*! mouse button got either pressed or released at given location */
	// virtual void mouseButtonRight (const vec2i &where, bool pressed) {}

protected:
	CameraFrame* cameraFrame;

	const float kbd_rotate_degrees{ 10.f };
	const float degrees_per_drag_fraction{ 150.f };
	const float pixels_per_move{ 10.f };
};