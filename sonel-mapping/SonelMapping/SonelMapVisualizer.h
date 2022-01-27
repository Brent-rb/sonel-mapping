#pragma once

// our own classes, partly shared between host and device
#include "../Cuda/CudaBuffer.h"
#include "Models/LaunchParams.h"
#include "../UI/Camera.h"
#include "Models/Model.h"
#include "OptixSetup.h"
#include "OptixScene.h"

class SonelMapVisualizer {
public:
	SonelMapVisualizer(
		const OptixSetup& optixSetup,
		OptixScene& cudaScene
	);

	void init();

    void setFrequencySize(uint32_t size);
	void setSonelArray(std::vector<std::vector<Sonel>>* newSonelArray);

	/*! render one frame */
	void render();

	/*! resize frame buffer to given resolution */
	void resize(const vec2i& newSize);

	/*! download the rendered color buffer */
	void downloadPixels(uint32_t h_pixels[]);
	void uploadSonelMapSnapshot();

	/*! set camera to render with */
	void setCamera(const Camera& camera);

protected:
	void createRenderModule();

	void createRenderRaygenPrograms();
	void createRenderMissPrograms();
	void createRenderHitgroupPrograms();
	void createRenderPipeline();
	void buildRenderSbt();

private:
	void buildRenderRaygenRecords();
	void buildRenderMissRecords();
	void buildRenderHitgroupRecords();

protected:
	const OptixSetup& optixSetup;
	OptixScene& cudaScene;
	std::vector<std::vector<Sonel>>* sonelArray;

	OptixPipeline renderPipeline;
	OptixPipelineCompileOptions renderPipelineCompileOptions = {};
	OptixPipelineLinkOptions renderPipelineLinkOptions = {};

	OptixModule renderModule;
	OptixModuleCompileOptions renderModuleCompileOptions = {};

	std::vector<OptixProgramGroup> renderRaygenPgs;
	CudaBuffer renderRaygenRecordsBuffer;
	std::vector<OptixProgramGroup> renderMissPgs;
	CudaBuffer renderMissRecordsBuffer;
	std::vector<OptixProgramGroup> renderHitgroupPgs;
	CudaBuffer renderHitgroupRecordsBuffer;
	OptixShaderBindingTable renderSbt = {};

	LaunchParams launchParams;
	CudaBuffer launchParamsBuffer;

	CudaBuffer colorBuffer;

	Camera lastSetCamera;

	uint32_t timestep;
};