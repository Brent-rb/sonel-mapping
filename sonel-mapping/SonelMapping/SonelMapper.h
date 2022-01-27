//
// Created by brent on 27/01/2022.
//

#ifndef SONEL_MAPPING_SONELMAPPER_H
#define SONEL_MAPPING_SONELMAPPER_H

#include "SmOptixProgram.h"
#include "../Cuda/CudaSonelMapperParams.h"

struct SonelMapperConfig {
	std::vector<SoundSource>& soundSources;
	float echogramDuration;
	float soundSpeed;
	float earSize;
	uint32_t frequencySize;
};

class SonelMapper: public SmOptixProgram<CudaSonelMapperParams, EmptyRecord, EmptyRecord, TriangleMeshSbtData> {
public:
	SonelMapper(
		const OptixSetup& optixSetup,
		const OptixScene& optixScene
	);

	void initialize(SonelMapperConfig config);

	void execute() override;
	std::vector<std::vector<Sonel>>* getSonelArrays();

	const SonelMapData& getSonelMapData() const;

protected:
	const char *getLaunchParamsName() override;

	void configureRaygenProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;
	void configureMissProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;
	void configureHitProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;

	void createHitRecords(std::vector<SmRecord<TriangleMeshSbtData>> &hitRecords) override;

	void downloadSonelDataForFrequency(uint32_t fIndex, uint32_t sourceIndex);

private:
	// Data
	SonelMapData sonelMap;
	SonelMapData* sonelMapDevicePtr;

	std::vector<std::vector<Sonel>> sonelArrays;
};


#endif //SONEL_MAPPING_SONELMAPPER_H
