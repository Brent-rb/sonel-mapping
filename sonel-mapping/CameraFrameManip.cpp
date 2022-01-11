#include "CameraFrameManip.h"
#include "CameraFrame.h"

CameraFrameManip::CameraFrameManip(CameraFrame* cameraFrame) : cameraFrame(cameraFrame) {

}

void CameraFrameManip::key(int key, int mods) {
	CameraFrame& fc = *cameraFrame;

	switch (key) {
		case '+':
		case '=':
			fc.motionSpeed *= 1.5f;
			std::cout << "# viewer: new motion speed is " << fc.motionSpeed << std::endl;
			break;
		case '-':
		case '_':
			fc.motionSpeed /= 1.5f;
			std::cout << "# viewer: new motion speed is " << fc.motionSpeed << std::endl;
			break;
		case 'C':
			std::cout << "(C)urrent camera:" << std::endl;
			std::cout << "- from :" << fc.position << std::endl;
			std::cout << "- poi  :" << fc.getPoi() << std::endl;
			std::cout << "- upVec:" << fc.upVector << std::endl;
			std::cout << "- frame:" << fc.frame << std::endl;
			break;
		case 'W':
			strafe(fc.get_forward() * fc.motionSpeed / 100.0f);
			break;
		case 'S':
			strafe(-fc.get_forward() * fc.motionSpeed / 100.0f);
			break;
		case 'A':
			strafe(-fc.get_right() * fc.motionSpeed / 100.0f);
			break;
		case 'D':
			strafe(fc.get_right() * fc.motionSpeed / 100.0f);
			break;
		default:
			break;
	}
}

void CameraFrameManip::strafe(const gdt::vec3f& howMuch) {
	cameraFrame->position += howMuch;
	cameraFrame->modified = true;
}

void CameraFrameManip::strafe(const gdt::vec2f& howMuch) {
	strafe(
		howMuch.x * cameraFrame->frame.vx
		- howMuch.y * cameraFrame->frame.vy
	);
}

void CameraFrameManip::mouseDragLeft(const gdt::vec2f& delta) {
	rotate(delta.x * degrees_per_drag_fraction, delta.y * degrees_per_drag_fraction);
}

void CameraFrameManip::mouseDragMiddle(const gdt::vec2f& delta) {
	strafe(delta * pixels_per_move * cameraFrame->motionSpeed);
}

void CameraFrameManip::mouseDragRight(const gdt::vec2f& delta) {
	move(delta.y * pixels_per_move * cameraFrame->motionSpeed);
}
