#include "GlfCameraWindow.h"
#include "FlyModeManip.h"
#include "InspectModeManip.h"


GlfCameraWindow::GlfCameraWindow(
	const std::string& title,
	const gdt::vec3f& camera_from,
	const gdt::vec3f& camera_at,
	const gdt::vec3f& camera_up,
	const float worldScale
): GLFWindow(title), cameraFrame(worldScale) {
	cameraFrame.setOrientation(camera_from, camera_at, camera_up);
	enableFlyMode();
	enableInspectMode();
}

inline void GlfCameraWindow::enableFlyMode() {
	flyModeManip = std::make_shared<FlyModeManip>(&cameraFrame);
	cameraFrameManip = flyModeManip;
}

inline void GlfCameraWindow::enableInspectMode() {
	inspectModeManip = std::make_shared<InspectModeManip>(&cameraFrame);
	cameraFrameManip = inspectModeManip;
}

void GlfCameraWindow::key(int key, int mods) {
	switch (key) {
		case 'f':
		case 'F':
			std::cout << "Entering 'fly' mode" << std::endl;
			if (flyModeManip) cameraFrameManip = flyModeManip;
			break;
		case 'i':
		case 'I':
			std::cout << "Entering 'inspect' mode" << std::endl;
			if (inspectModeManip) cameraFrameManip = inspectModeManip;
			break;
		default:
			if (cameraFrameManip)
				cameraFrameManip->key(key, mods);
	}
}

/*! callback that window got resized */
void GlfCameraWindow::mouseMotion(const vec2i& newPos) {
	vec2i windowSize;
	glfwGetWindowSize(handle, &windowSize.x, &windowSize.y);

	if (isPressed.leftButton && cameraFrameManip)
		cameraFrameManip->mouseDragLeft(gdt::vec2f(newPos - lastMousePos) / gdt::vec2f(windowSize));
	if (isPressed.rightButton && cameraFrameManip)
		cameraFrameManip->mouseDragRight(gdt::vec2f(newPos - lastMousePos) / gdt::vec2f(windowSize));
	if (isPressed.middleButton && cameraFrameManip)
		cameraFrameManip->mouseDragMiddle(gdt::vec2f(newPos - lastMousePos) / gdt::vec2f(windowSize));
	lastMousePos = newPos;
	/* empty - to be subclassed by user */
}

/*! callback that window got resized */
void GlfCameraWindow::mouseButton(int button, int action, int mods) {
	const bool pressed = (action == GLFW_PRESS);
	switch (button) {
		case GLFW_MOUSE_BUTTON_LEFT:
			isPressed.leftButton = pressed;
			break;
		case GLFW_MOUSE_BUTTON_MIDDLE:
			isPressed.middleButton = pressed;
			break;
		case GLFW_MOUSE_BUTTON_RIGHT:
			isPressed.rightButton = pressed;
			break;
	}
	lastMousePos = getMousePos();
}