#pragma once
#include "gdt/math/vec.h"
#include "gdt/math/box.h"
#include <map>
#include <vector>
#include "AabbItem.h"

class Sonel: public AabbItem {
public:
	Sonel(): Sonel(gdt::vec3f(0.0f), gdt::vec3f(0.0f), 0.0f, 0.0f, 0, 0) {

	}

	Sonel(gdt::vec3f position, gdt::vec3f incidence,
		  float energy, float time,
		  uint32_t frequency, uint16_t frequencyIndex): AabbItem(), position(position), incidence(incidence), energy(energy), time(time), frequency(frequency), frequencyIndex(frequencyIndex) {

	}

	gdt::vec3f getPosition() const override {
		return position;
	}

	float getRadius() const override {
		return 0.15f;
	}

public:
	gdt::vec3f position;
	gdt::vec3f incidence;
	float energy;
	float time;
	float distance;
	uint32_t frequency;
    uint16_t frequencyIndex;
};