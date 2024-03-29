//
// Created by brent on 27/01/2022.
//

#ifndef SONEL_MAPPING_SONELMAPPER_H
#define SONEL_MAPPING_SONELMAPPER_H

#include "SmOptixProgram.h"
#include "../Cuda/CudaSonelMapperParams.h"
#include "models/SimulationData.h"

struct SonelMapperConfig {
	std::vector<SoundSource>& soundSources;
	float echogramDuration;
	float timestep;
	float soundSpeed;
};

class SonelMapper: public SmOptixProgram<CudaSonelMapperParams, EmptyRecord, EmptyRecord, SmSbtData> {
public:
	SonelMapper(
		const OptixSetup& optixSetup,
		OptixScene& optixScene
	);

	void initialize(SonelMapperConfig config);

	void execute(std::ofstream& timingFile) override;
	std::vector<Sonel>* getSonelArray();

	SimulationData& getSimulationData();

protected:
	const char *getLaunchParamsName() override;

	void configureRaygenProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;
	void configureMissProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;
	void configureHitProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;

	void addHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords) override;

	void downloadSonelDataForFrequency(uint32_t fIndex, uint32_t sourceIndex);

private:
	// Data
	float maxTime;
	SimulationData simulationData;
	SimulationData* sonelMapDevicePtr;

	std::vector<Sonel> sonelArray;
};


#endif //SONEL_MAPPING_SONELMAPPER_H
