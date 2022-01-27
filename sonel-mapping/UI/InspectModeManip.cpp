#include "gdt/math/AffineSpace.h"
#include "InspectModeManip.h"
#include "CameraFrame.h"

InspectModeManip::InspectModeManip(CameraFrame* cameraFrame) : CameraFrameManip(cameraFrame) {}

/*! helper function: rotate camera frame by given degrees, then
	make sure the frame, poidistance etc are all properly set,
	the widget gets notified, etc */
void InspectModeManip::rotate(const float deg_u, const float deg_v) {
	float rad_u = -M_PI / 180.f * deg_u;
	float rad_v = -M_PI / 180.f * deg_v;

	CameraFrame& fc = *cameraFrame;

	const gdt::vec3f poi = fc.getPoi();
	fc.frame
		= gdt::linear3f::rotate(fc.frame.vy, rad_u)
		* gdt::linear3f::rotate(fc.frame.vx, rad_v)
		* fc.frame;

	if (fc.forceUp) fc.forceUpFrame();

	fc.position = poi + fc.poiDistance * fc.frame.vz;
	fc.modified = true;
}

/*! helper function: move forward/backwards by given multiple of
	motion speed, then make sure the frame, poidistance etc are
	all properly set, the widget gets notified, etc */
void InspectModeManip::move(const float step) {
	const gdt::vec3f poi = cameraFrame->getPoi();
	// inspectmode can't get 'beyond' the look-at point:
	const float minReqDistance = 0.1f * cameraFrame->motionSpeed;
	cameraFrame->poiDistance = gdt::max(minReqDistance, cameraFrame->poiDistance - step);
	cameraFrame->position = poi + cameraFrame->poiDistance * cameraFrame->frame.vz;
	cameraFrame->modified = true;
}