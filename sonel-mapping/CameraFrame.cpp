
#include "CameraFrame.h"

// common gdt helper tools
#include "gdt/math/AffineSpace.h"

CameraFrame::CameraFrame(const float worldScale): motionSpeed(worldScale) {

}

gdt::vec3f CameraFrame::getPoi() const {
	return position - poiDistance * frame.vz;
}

void CameraFrame::setOrientation(
	const gdt::vec3f& origin,
	const gdt::vec3f& interest,
	const gdt::vec3f& up
) {
	position = origin;
	upVector = up;
	
	frame.vz = (interest == origin ? gdt::vec3f(0, 0, 1) : -gdt::normalize(interest - origin)); /* negative because we use NEGATIZE z axis */
	frame.vx = gdt::cross(up, frame.vz);
	
	if (gdt::dot(frame.vx, frame.vx) < 1e-8f) {
		frame.vx = gdt::vec3f(0, 1, 0);
	}
	else {
		frame.vx = gdt::normalize(frame.vx);
	}

	// frame.vx
	//   = (fabs(dot(up,frame.vz)) < 1e-6f)
	//   ? gdt::vec3f(0,1,0)
	//   : normalize(cross(up,frame.vz));
	frame.vy = gdt::normalize(gdt::cross(frame.vz, frame.vx));
	poiDistance = length(interest - origin);
	forceUpFrame();
}

void CameraFrame::forceUpFrame() {
	// frame.vz remains unchanged
	if (fabsf(gdt::dot(frame.vz, upVector)) < 1e-6f)
		// looking along upvector; not much we can do here ...
		return;
	frame.vx = gdt::normalize(gdt::cross(upVector, frame.vz));
	frame.vy = gdt::normalize(gdt::cross(frame.vz, frame.vx));
	modified = true;
}

void CameraFrame::setUpVector(const gdt::vec3f& up) {
	upVector = up; forceUpFrame();
}