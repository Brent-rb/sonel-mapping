//
// Created by brent on 05/02/2022.
//


#ifndef SONEL_MAPPING_AABBITEM_H
#define SONEL_MAPPING_AABBITEM_H

#include "gdt/math/vec.h"

class AabbItem {
public:
	virtual gdt::vec3f getPosition() const = 0;
	virtual float getRadius() const = 0;
};

#endif //SONEL_MAPPING_AABBITEM_H
