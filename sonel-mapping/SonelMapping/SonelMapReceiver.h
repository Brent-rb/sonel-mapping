//
// Created by brent on 27/01/2022.
//

#ifndef SONEL_MAPPING_SONELMAPRECEIVER_H
#define SONEL_MAPPING_SONELMAPRECEIVER_H

#include "SmOptixProgram.h"
#include "Models/SonelReceiverParams.h"
#include "../UI/Camera.h"

struct SonelMapReceiverConfig {
	uint32_t rayAmount = 0;
	float soundSpeed = 0.0f;
	float duration = 0.0f;
	float timestep = 0.0f;
	uint32_t frequencySize = 0;
	uint32_t timestepSize = 0;
};

class SonelMapReceiver: public SmOptixProgram<SonelReceiverParams, EmptyRecord, EmptyRecord, TriangleMeshSbtData> {
public:
	SonelMapReceiver(
		const OptixSetup& optixSetup,
		OptixScene& optixScene
	);

	void initialize(SonelMapReceiverConfig config);

	void execute() override;

	void setCamera(const Camera& camera);
	void setSonels(std::vector<Sonel>* sonels);

protected:
	const char *getLaunchParamsName() override;

	void configureRaygenProgram(
			uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
	) override;

	void configureMissProgram(
			uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
	) override;

	void
	configureHitProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;

	void addHitRecords(std::vector<SmRecord<TriangleMeshSbtData>> &hitRecords) override;

	void configureModuleCompileOptions(OptixModuleCompileOptions &compileOptions) override;

	void configurePipelineCompileOptions(OptixPipelineCompileOptions &pipelineOptions) override;

	void configurePipelineLinkOptions(OptixPipelineLinkOptions &pipelineLinkOptions) override;

private:
	SonelMapReceiverConfig config;
	CudaBuffer energyBuffer;
	std::vector<Sonel>* sonels;

	uint32_t bufferSize;
};


#endif //SONEL_MAPPING_SONELMAPRECEIVER_H
