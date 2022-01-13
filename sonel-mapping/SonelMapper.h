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
#include "SonelMap.h"
#include "CudaSonelMapperParams.h"
#include "OctTree.h"
#include "OptixSetup.h"
#include "OptixScene.h"

struct SonelMapperConfig {
	std::vector<SoundSource>& soundSources;
	float echogramDuration;
	float soundSpeed;
	float earSize;
	uint32_t frequencySize;
};

class SonelMapper {
	public:
		/*! constructor - performs all setup, including initializing
		  optix, creates module, pipeline, programs, SBT, etc. */
		SonelMapper(
			const OptixSetup& optixSetup,
			const OptixScene& cudaScene,
			SonelMapperConfig config
		);

		
		void calculate();
		std::vector<OctTree<Sonel>>* getSonelMap();

	protected:
		void createSonelModule();

		void createSonelRaygenPrograms();
		void createSonelMissPrograms();
		void createSonelHitgroupPrograms();
		void createSonelPipeline();

		void buildSonelSbt();

		void launchOptix(SoundFrequency& frequency, uint32_t sourceIndex);
		void launchOptixForFrequency(uint32_t fIndex);
		void downloadSonelDataForFrequency(uint32_t fIndex, uint32_t sourceIndex);

	private:
		void buildSonelRaygenRecords();
		void buildSonelMissRecords();
		void buildSonelHitgroupRecords();

	protected:
		const OptixSetup& optixSetup;
		const OptixScene& cudaScene;

		// Pipeline
		OptixPipeline sonelPipeline;
		OptixPipelineCompileOptions sonelPipelineCompileOptions = {};
		OptixPipelineLinkOptions sonelPipelineLinkOptions = {};

		// Module
		OptixModule sonelModule;
		OptixModuleCompileOptions sonelModuleCompileOptions = {};

		// Programs
		std::vector<OptixProgramGroup> sonelRaygenPgs;
		CUDABuffer sonelRaygenRecordsBuffer;
		std::vector<OptixProgramGroup> sonelMissPgs;
		CUDABuffer sonelMissRecordsBuffer;
		std::vector<OptixProgramGroup> sonelHitgroupPgs;
		CUDABuffer sonelHitgroupRecordsBuffer;
		OptixShaderBindingTable sonelSbt = {};

		// Data
		SonelMapData sonelMap;
		SonelMapData* sonelMapDevicePtr;

		CudaSonelMapperParams launchParams;
		CudaSonelMapperParams* launchParamsDevicePtr;

		uint32_t frequencyIndex;
		uint32_t frequencySize;
		bool hasCalculatedSonelMap = false;

		std::vector<OctTree<Sonel>> octTrees;
};
