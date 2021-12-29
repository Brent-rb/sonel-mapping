#pragma once

#include "gdt/math/vec.h"
#include "gdt/math/AffineSpace.h"

struct CameraFrame {
	CameraFrame(const float worldScale);

	gdt::vec3f getPoi() const;

	/*! re-compute all orientation related fields from given
		'user-style' camera parameters */
	void setOrientation(
		const gdt::vec3f& origin,
		const gdt::vec3f& interest,
		const gdt::vec3f& up
	);
	

	/*! tilt the frame around the z axis such that the y axis is "facing upwards" */
	void forceUpFrame();

	void setUpVector(const gdt::vec3f& up);

	inline float computeStableEpsilon(float f) const {
		return abs(f) * float(1. / (1 << 21));
	}

	inline float computeStableEpsilon(const gdt::vec3f v) const {
		return gdt::max(
			gdt::max(
				computeStableEpsilon(v.x),
				computeStableEpsilon(v.y)
			),
			computeStableEpsilon(v.z)
		);
	}

	inline gdt::vec3f get_from() const { return position; }
	inline gdt::vec3f get_at() const { return getPoi(); }
	inline gdt::vec3f get_up() const { return upVector; }

	gdt::linear3f frame{ gdt::one };
	gdt::vec3f position{ 0,-1,0 };

	/*! distance to the 'point of interst' (poi); e.g., the point we
		will rotate around */
	float poiDistance{ 1.f };

	gdt::vec3f upVector{ 0,1,0 };

	/* if set to true, any change to the frame will always use to
		upVector to 'force' the frame back upwards; if set to false,
		the upVector will be ignored */
	bool forceUp{ true };

	/*! multiplier how fast the camera should move in world space
		for each unit of "user specifeid motion" (ie, pixel
		count). Initial value typically should depend on the world
		size, but can also be adjusted. This is actually something
		that should be more part of the manipulator widget(s), but
		since that same value is shared by multiple such widgets
		it's easiest to attach it to the camera here ...*/
	float motionSpeed{ 1.f };

	/*! gets set to true every time a manipulator changes the camera
		values */
	bool modified{ true };
};
