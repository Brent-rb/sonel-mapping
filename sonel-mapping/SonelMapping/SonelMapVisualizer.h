#pragma once
#ifndef SONELMAPVISUALIZER_H
#define SONELMAPVISUALIZER_H

// our own classes, partly shared between host and device
#include "../Cuda/CudaBuffer.h"
#include "Models/LaunchParams.h"
#include "../UI/Camera.h"
#include "Models/Model.h"
#include "OptixSetup.h"
#include "OptixScene.h"
#include "SmOptixProgram.h"
#include "Models/TriangleMeshSbtData.h"

class SonelMapVisualizer: public SmOptixProgram<LaunchParams, EmptyRecord, EmptyRecord, TriangleMeshSbtData> {
public:
	SonelMapVisualizer(
		const OptixSetup& optixSetup,
		OptixScene& cudaScene
	);

	void initialize();

    void setFrequencySize(uint32_t size);
	void setSonelArray(std::vector<std::vector<Sonel>>* newSonelArray);

	void execute() override;

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

	void addHitRecords(std::vector<SmRecord<TriangleMeshSbtData>> &hitRecords) override;

	void configureModuleCompileOptions(OptixModuleCompileOptions &compileOptions) override;
	void configurePipelineCompileOptions(OptixPipelineCompileOptions &pipelineOptions) override;
	void configurePipelineLinkOptions(OptixPipelineLinkOptions &pipelineLinkOptions) override;

protected:
	OptixScene& cudaScene;
	std::vector<std::vector<Sonel>>* sonelArray;

	CudaBuffer colorBuffer;

	Camera lastSetCamera;

	uint32_t timestep;
};

#endif