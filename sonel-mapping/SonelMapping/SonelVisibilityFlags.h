//
// Created by brent on 27/01/2022.
//

#ifndef SONEL_MAPPING_OPTIXVISIBILITYMASKS_H
#define SONEL_MAPPING_OPTIXVISIBILITYMASKS_H

enum SonelVisibilityFlags {
	GEOMETRY_VISIBLE = 1,
	SONELS_VISIBLE = 2,
	SOUND_SOURCES_VISIBLE = 4
};

inline SonelVisibilityFlags operator|(SonelVisibilityFlags a, SonelVisibilityFlags b)
{
	return static_cast<SonelVisibilityFlags>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

#endif //SONEL_MAPPING_OPTIXVISIBILITYMASKS_H
