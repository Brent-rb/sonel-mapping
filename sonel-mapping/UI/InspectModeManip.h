#pragma once

#include "CameraFrameManip.h"

struct CameraFrame;

// ------------------------------------------------------------------
/*! camera manipulator with the following traits

	- there is a "point of interest" (POI) that the camera rotates
	around.  (we track this via poiDistance, the point is then
	thta distance along the fame's z axis)

	- we can restrict the minimum and maximum distance camera can be
	away from this point

	- we can specify a max bounding box that this poi can never
	exceed (validPoiBounds).

	- we can restrict whether that point can be moved (by using a
	single-point valid poi bounds box

	- left drag rotates around the object

	- right drag moved forward, backward (within min/max distance
	bounds)

	- middle drag strafes left/right/up/down (within valid poi
	bounds)

*/
struct InspectModeManip: public CameraFrameManip {
	InspectModeManip(CameraFrame* cameraFrame);

private:
	/*! helper function: rotate camera frame by given degrees, then
		make sure the frame, poidistance etc are all properly set,
		the widget gets notified, etc */
	virtual void rotate(const float deg_u, const float deg_v) override;

	/*! helper function: move forward/backwards by given multiple of
		motion speed, then make sure the frame, poidistance etc are
		all properly set, the widget gets notified, etc */
	virtual void move(const float step) override;
};