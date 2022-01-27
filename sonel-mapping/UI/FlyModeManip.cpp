#include "FlyModeManip.h"
#include "CameraFrame.h"
#include "gdt/math/AffineSpace.h"

FlyModeManip::FlyModeManip(CameraFrame* cameraFrame): CameraFrameManip(cameraFrame) {

}

/*! helper function: rotate camera frame by given degrees, then
	make sure the frame, poidistance etc are all properly set,
	the widget gets notified, etc */
void FlyModeManip::rotate(const float deg_u, const float deg_v) {
	float rad_u = -M_PI / 180.f * deg_u;
	float rad_v = -M_PI / 180.f * deg_v;

	CameraFrame& fc = *cameraFrame;

	//const gdt::vec3f poi  = fc.getPOI();
	fc.frame
		= gdt::linear3f::rotate(fc.frame.vy, rad_u)
		* gdt::linear3f::rotate(fc.frame.vx, rad_v)
		* fc.frame;

	if (fc.forceUp) fc.forceUpFrame();

	fc.modified = true;
}

/*! helper function: move forward/backwards by given multiple of
	motion speed, then make sure the frame, poidistance etc are
	all properly set, the widget gets notified, etc */
void FlyModeManip::move(const float step) {
	cameraFrame->position += step * cameraFrame->frame.vz;
	cameraFrame->modified = true;
}