//
// Created by brent on 27/01/2022.
//

#ifndef SONEL_MAPPING_SONELMAPRECEIVER_H
#define SONEL_MAPPING_SONELMAPRECEIVER_H

#include "SmOptixProgram.h"
#include "Models/SonelReceiverParams.h"
#include "../UI/Camera.h"
#include "Models/SimulationData.h"
#include "Models/AbsorptionData.h"

struct SonelMapReceiverConfig {
	uint32_t rayAmount = 0;
	float soundSpeed = 0.0f;
	float duration = 0.0f;
	float timestep = 0.0f;
	uint32_t frequencySize = 0;
	uint32_t timestepSize = 0;
	uint16_t maxSonels = 100;
	SimulationData* simulationData = nullptr;
	AbsorptionData* absorptionData = nullptr;
};

class SonelMapReceiver: public SmOptixProgram<SonelReceiverParams, EmptyRecord, EmptyRecord, SmSbtData> {
public:
	SonelMapReceiver(
		const OptixSetup& optixSetup,
		OptixScene& optixScene
	);

	void initialize(SonelMapReceiverConfig newConfig);

	void execute(std::ofstream& timingFile) override;

	void setCamera(const Camera& camera);
	void setSonels(std::vector<Sonel>* sonels);

protected:
	const char *getLaunchParamsName() override;

	void configureRaygenProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;
	void configureMissProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;
	void configureHitProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;

	void addHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords) override;
	void addGeometryHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords);
	void addSonelHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords);
	void addSoundSourceHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords);

	void configureModuleCompileOptions(OptixModuleCompileOptions &compileOptions) override;
	void configurePipelineCompileOptions(OptixPipelineCompileOptions &pipelineOptions) override;
	void configurePipelineLinkOptions(OptixPipelineLinkOptions &pipelineLinkOptions) override;

	void simulate();
	void configureScene();
	void initEchogram();
	void addLaunchToEchogram();
	void writeEchogram();
private:
	SonelMapReceiverConfig config;

	CudaBuffer entriesBuffer;
	CudaBuffer hitBuffer;
	CudaBuffer absorptionBuffer;

	std::vector<Sonel>* sonels;
	std::vector<std::vector<float>> echogram;

	uint32_t bufferSize;
	uint32_t highestTimestep = 0;
};


#endif //SONEL_MAPPING_SONELMAPRECEIVER_H
