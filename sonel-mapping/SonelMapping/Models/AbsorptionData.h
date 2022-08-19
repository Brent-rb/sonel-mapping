#pragma once

class AbsorptionData {
public:
	AbsorptionData(uint16_t humidity) : humidity(humidity), absorptions(nullptr) {

	}

	void setAbsorptions(const uint32_t* frequencies, uint32_t frequencySize) {
		absorptions = new float[frequencySize];
		std::map<uint16_t, float>& humidtyAbsorption = airAbsorptionMap[humidity];
		for(uint16_t i = 0; i < frequencySize; i++) {
			absorptions[i] = humidtyAbsorption[frequencies[i]];
		}

		absorptionSize = frequencySize;
	}

public:
	uint16_t humidity;
	uint16_t absorptionSize;
	float* absorptions;

	std::map<uint16_t, std::map<uint16_t, float>> airAbsorptionMap = {
		{ 40, {
			 { 63, 0.013f },
			 { 125, 0.05f },
			 { 250, 0.09f },
			 { 500, 0.6f },
			 { 1000, 1.07f },
			 { 2000, 2.58f },
			 { 4000, 5.05f }
			}
		}, {50, {
		   { 63, 0.009f },
		   { 125, 0.045f },
		   { 250, 0.08f },
		   { 500, 0.63f },
		   { 1000, 1.08f },
		   { 2000, 2.28f },
		   { 4000, 4.2f }
	   }}, {60, {
		   { 63, 0.008f },
		   { 125, 0.025f },
		   { 250, 0.06f },
		   { 500, 0.64f },
		   { 1000, 1.11f },
		   { 2000, 2.14f },
		   { 4000, 3.72f }
	   }}, {70, {
		   { 63, 0.006f },
		   { 125, 0.02f },
		   { 250, 0.05f },
		   { 500, 0.64f },
		   { 1000, 1.15f },
		   { 2000, 2.08f },
		   { 4000, 3.45f }
	   }}
	};
};