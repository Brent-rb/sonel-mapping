// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

// our own classes, partly shared between host and device
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Camera.h"
#include "Model.h"
#include "SoundSource.h"

class SonelMapper {
	public:
		/*! constructor - performs all setup, including initializing
		  optix, creates module, pipeline, programs, SBT, etc. */
		SonelMapper(const Model* model, const QuadLight& light, SoundSource soundSource, float echogramDuration, float soundSpeed, float earSize);

		/*! render one frame */
		void render();

		/*! resize frame buffer to given resolution */
		void resize(const vec2i& newSize);

		/*! download the rendered color buffer */
		void downloadPixels(uint32_t h_pixels[]);
		void downloadSonelMap(Sonel sonels[]);
		void uploadSonelMapSnapshot(int index);

		/*! set camera to render with */
		void setCamera(const Camera& camera);

	protected:
		// ------------------------------------------------------------------
		// internal helper functions
		// ------------------------------------------------------------------

		/*! helper function that initializes optix and checks for errors */
		void initOptix();

		/*! creates and configures a optix device context (in this simple
		  example, only for the primary GPU device) */
		void createContext();

		/*! creates the module that contains all the programs we are going
		  to use. in this simple example, we use a single module from a
		  single .cu file, using a single embedded ptx string */
		void createRenderModule();
		void createSonelModule();

		/*! does all setup for the raygen program(s) we are going to use */
		void createRenderRaygenPrograms();
		void createSonelRaygenPrograms();

		/*! does all setup for the miss program(s) we are going to use */
		void createRenderMissPrograms();
		void createSonelMissPrograms();

		/*! does all setup for the hitgroup program(s) we are going to use */
		void createRenderHitgroupPrograms();
		void createSonelHitgroupPrograms();

		/*! assembles the full pipeline of all programs */
		void createRenderPipeline();
		void createSonelPipeline();

		/*! constructs the shader binding table */
		void buildRenderSbt();
		void buildSonelSbt();

		/*! build an acceleration structure for the given triangle mesh */
		OptixTraversableHandle buildAccel();

		/*! upload textures, and create cuda texture objects for them */
		void createTextures();

	private:
		void buildRenderRaygenRecords();
		void buildSonelRaygenRecords();
		void buildRenderMissRecords();
		void buildSonelMissRecords();
		void buildRenderHitgroupRecords();
		void buildSonelHitgroupRecords();
		void initSonelBuffer();

	protected:
		/*! @{ CUDA device context and stream that optix pipeline will run
			on, as well as device properties for this device */
		CUcontext cudaContext;
		CUstream stream;
		cudaDeviceProp deviceProps;
		/*! @} */

		//! the optix context that our pipeline will run in.
		OptixDeviceContext optixContext;

		/*! @{ the pipeline we're building */
		OptixPipeline renderPipeline;
		OptixPipelineCompileOptions renderPipelineCompileOptions = {};
		OptixPipelineLinkOptions renderPipelineLinkOptions = {};
		/*! @} */

		/*! @{ the module that contains out device programs */
		OptixModule renderModule;
		OptixModuleCompileOptions renderModuleCompileOptions = {};
		/* @} */

		/*! vector of all our program(group)s, and the SBT built around
			them */
		std::vector<OptixProgramGroup> renderRaygenPgs;
		CUDABuffer renderRaygenRecordsBuffer;
		std::vector<OptixProgramGroup> renderMissPgs;
		CUDABuffer renderMissRecordsBuffer;
		std::vector<OptixProgramGroup> renderHitgroupPgs;
		CUDABuffer renderHitgroupRecordsBuffer;
		OptixShaderBindingTable renderSbt = {};

		/*! @{ the pipeline we're building */
		OptixPipeline sonelPipeline;
		OptixPipelineCompileOptions sonelPipelineCompileOptions = {};
		OptixPipelineLinkOptions sonelPipelineLinkOptions = {};
		/*! @} */

		/*! @{ the module that contains out device programs */
		OptixModule sonelModule;
		OptixModuleCompileOptions sonelModuleCompileOptions = {};
		/* @} */

		/*! vector of all our program(group)s, and the SBT built around
			them */
		std::vector<OptixProgramGroup> sonelRaygenPgs;
		CUDABuffer sonelRaygenRecordsBuffer;
		std::vector<OptixProgramGroup> sonelMissPgs;
		CUDABuffer sonelMissRecordsBuffer;
		std::vector<OptixProgramGroup> sonelHitgroupPgs;
		CUDABuffer sonelHitgroupRecordsBuffer;
		OptixShaderBindingTable sonelSbt = {};

		/*! @{ our launch parameters, on the host, and the buffer to store
			them on the device */
		LaunchParams launchParams;
		CUDABuffer launchParamsBuffer;
		/*! @} */

		CUDABuffer colorBuffer;
		CUDABuffer sonelMapBuffer;

		/*! the camera we are to render with. */
		Camera lastSetCamera;

		/*! the model we are going to trace rays against */
		const Model* model;

		/*! @{ one buffer per input mesh */
		std::vector<CUDABuffer> vertexBuffer;
		std::vector<CUDABuffer> normalBuffer;
		std::vector<CUDABuffer> texcoordBuffer;
		std::vector<CUDABuffer> indexBuffer;
		/*! @} */

		//! buffer that keeps the (final, compacted) accel structure
		CUDABuffer asBuffer;

		/*! @{ one texture object and pixel array per used texture */
		std::vector<cudaArray_t> textureArrays;
		std::vector<cudaTextureObject_t> textureObjects;
		/*! @} */

		SonelMap* sonelMap;
		int sonelMapIndex;

		bool hasCalculatedSonelMap = false;
};
