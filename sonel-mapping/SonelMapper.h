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
#include "SonelMap.h"
#include "CudaSonelMapperParams.h"
#include "OctTree.h"

class SonelMapper {
	public:
		/*! constructor - performs all setup, including initializing
		  optix, creates module, pipeline, programs, SBT, etc. */
		SonelMapper(const Model* model, const std::vector<SoundSource>& soundSources, float echogramDuration, float soundSpeed, float earSize, uint32_t frequencySize);

		/*! render one frame */
		void calculate();

	protected:
		// ------------------------------------------------------------------
		// internal helper functions
		// ------------------------------------------------------------------

		/*! helper function that initializes optix and checks for errors */
		void initOptix();

		/*! creates and configures a optix device context (in this simple
		  example, only for the primary GPU device) */
		void createContext();

		void createSonelModule();

		void createSonelRaygenPrograms();
		void createSonelMissPrograms();
		void createSonelHitgroupPrograms();
		void createSonelPipeline();

		void buildSonelSbt();

		/*! build an acceleration structure for the given triangle mesh */
		OptixTraversableHandle buildAccel();

		/*! upload textures, and create cuda texture objects for them */
		void createTextures();

	private:
		void buildSonelRaygenRecords();
		void buildSonelMissRecords();
		void buildSonelHitgroupRecords();

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

		CUDABuffer sonelMapBuffer;

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

		SonelMapData sonelMap;
		SonelMapData* sonelMapDevicePtr;

		CudaSonelMapperParams launchParams;
		CudaSonelMapperParams* launchParamsDevicePtr;

		uint32_t frequencyIndex;
		uint32_t frequencySize;
		bool hasCalculatedSonelMap = false;

		std::vector<OctTree<Sonel>> octTrees;
};
