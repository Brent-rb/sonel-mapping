#pragma once
#ifndef SONELMAPVISUALIZER_H
#define SONELMAPVISUALIZER_H

// our own classes, partly shared between host and device
#include "../Cuda/CudaBuffer.h"
#include "Models/SonelVisualizerParams.h"
#include "../UI/Camera.h"
#include "Models/Model.h"
#include "OptixSetup.h"
#include "OptixScene.h"
#include "SmOptixProgram.h"
#include "Models/SmSbtData.h"

class SonelMapVisualizer: public SmOptixProgram<SonelVisualizerParams, EmptyRecord, EmptyRecord, SmSbtData> {
public:
	SonelMapVisualizer(
		const OptixSetup& optixSetup,
		OptixScene& cudaScene,
        float timestep,
        float sonelRadius
	);

	void initialize();

    void setFrequencySize(uint32_t size);
	void setSonelArray(std::vector<Sonel>* newSonelArray);

	void execute(std::ofstream& timingFile) override;
    void nextFrame();
    void previousFrame();

	/*! resize frame buffer to given resolution */
	void resize(const vec2i& newSize);

	/*! download the rendered color buffer */
	void downloadPixels(uint32_t h_pixels[]);
	void uploadSonelMapSnapshot();

	/*! set camera to render with */
	void setCamera(const Camera& camera);

protected:
	const char *getLaunchParamsName() override;

	void configureRaygenProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;
	void configureMissProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;
	void configureHitProgram(uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc) override;

	void addHitRecords(std::vector<SmRecord<SmSbtData>> &hitRecords) override;

	void configureModuleCompileOptions(OptixModuleCompileOptions &compileOptions) override;
	void configurePipelineCompileOptions(OptixPipelineCompileOptions &pipelineOptions) override;
	void configurePipelineLinkOptions(OptixPipelineLinkOptions &pipelineLinkOptions) override;

protected:
	std::vector<Sonel>* sonelArray;
	CudaBuffer colorBuffer;
	Camera lastSetCamera;
	uint32_t timestep = 0;
};

#endif