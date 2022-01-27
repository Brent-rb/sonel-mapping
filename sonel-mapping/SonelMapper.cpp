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

#include <vector>

#include "SonelMapper.h"
#include "CudaHelper.h"
#include "TriangleMeshSbtData.h"
#include "Model.h"

extern "C" char embedded_mapper_code[];

enum SonelMapperRayTypes { DefaultRay = 0, RaySize };

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	// just a dummy value - later examples will use more interesting
	// data here
	void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	// just a dummy value - later examples will use more interesting
	// data here
	void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	TriangleMeshSbtData data;
};

SonelMapper::SonelMapper(
	const OptixSetup& optixSetup,
	const OptixScene& cudaScene
): optixSetup(optixSetup), cudaScene(cudaScene), 
   frequencyIndex(0), frequencySize(0) {
	
}

void SonelMapper::init(SonelMapperConfig config) {
	frequencySize = config.frequencySize;
	sonelMap.setSoundSources(config.soundSources);
	sonelMap.duration = config.echogramDuration;
	sonelMap.soundSpeed = config.soundSpeed;
	sonelMap.timestep = 0.01;

	sonelMapDevicePtr = sonelMap.cudaCreate();
	CudaSonelMapperParams* paramsDevice;
	cudaMalloc(&paramsDevice, sizeof(CudaSonelMapperParams));

	launchParams.localFrequencyIndex = 0;
	launchParams.sonelMapData = sonelMapDevicePtr;
	launchParamsDevicePtr = paramsDevice;

	std::cout << "[SonelMapper] Creating sonel module." << std::endl;
	createSonelModule();

	std::cout << "[SonelMapper] Creating sonel raygen programs." << std::endl;
	createSonelRaygenPrograms();

	std::cout << "[SonelMapper] Creating sonel miss programs." << std::endl;
	createSonelMissPrograms();

	std::cout << "[SonelMapper] Creating sonel hit programs." << std::endl;
	createSonelHitgroupPrograms();

	launchParams.traversable = cudaScene.getGeoTraversable();

	std::cout << "[SonelMapper] Creating sonel pipeline." << std::endl;
	createSonelPipeline();

	std::cout << "[SonelMapper] Building SBT." << std::endl;
	buildSonelSbt();

	std::cout << GDT_TERMINAL_GREEN;
	std::cout << "[SonelMapper] Initialized." << std::endl;
	std::cout << GDT_TERMINAL_DEFAULT;
}

/*! creates the module that contains all the programs we are going
	to use. in this simple example, we use a single module from a
	single .cu file, using a single embedded ptx string */
void SonelMapper::createSonelModule() {
	sonelModuleCompileOptions.maxRegisterCount = 50;
	sonelModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	sonelModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	sonelPipelineCompileOptions = {};
	sonelPipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	sonelPipelineCompileOptions.usesMotionBlur = false;
	sonelPipelineCompileOptions.numPayloadValues = 2;
	sonelPipelineCompileOptions.numAttributeValues = 2;
	sonelPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	sonelPipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

	sonelPipelineLinkOptions.maxTraceDepth = 8;

	const std::string ptxCode = embedded_mapper_code;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	optixCheck(
		optixModuleCreateFromPTX(
			optixSetup.getOptixContext(), 
			&sonelModuleCompileOptions, 
			&sonelPipelineCompileOptions,
			ptxCode.c_str(), 
			ptxCode.size(), 
			log, 
			&sizeof_log, 
			&sonelModule
		),
		"SonelMapper",
		"Failed to create SonelMapper module."
	);

	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the raygen program(s) we are going to use */
void SonelMapper::createSonelRaygenPrograms() {
	// we do a single ray gen program in this example:
	sonelRaygenPgs.resize(SonelMapperRayTypes::RaySize);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = sonelModule;
	pgDesc.raygen.entryFunctionName = "__raygen__generateSonelMap";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	optixCheck(
		optixProgramGroupCreate(
			optixSetup.getOptixContext(),
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&sonelRaygenPgs[SonelMapperRayTypes::DefaultRay]
		),
		"SonelMapper",
		"Failed to create raygen programs."
	);
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the miss program(s) we are going to use */
void SonelMapper::createSonelMissPrograms() {
	// we do a single ray gen program in this example:
	sonelMissPgs.resize(SonelMapperRayTypes::RaySize);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = sonelModule;

	// ------------------------------------------------------------------
	// radiance rays
	// ------------------------------------------------------------------
	pgDesc.miss.entryFunctionName = "__miss__sonelRadiance";

	optixCheck(
		optixProgramGroupCreate(
			optixSetup.getOptixContext(), 
			&pgDesc, 
			1, 
			&pgOptions, 
			log,
			&sizeof_log,
			&sonelMissPgs[SonelMapperRayTypes::DefaultRay]
		),
		"SonelMapper",
		"Failed to create miss programs."
	);

	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void SonelMapper::createSonelHitgroupPrograms() {
	// for this simple example, we set up a single hit group
	sonelHitgroupPgs.resize(SonelMapperRayTypes::RaySize);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = sonelModule;
	pgDesc.hitgroup.moduleAH = sonelModule;

	// -------------------------------------------------------
	// radiance rays
	// -------------------------------------------------------
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__sonelRadiance";
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__sonelRadiance";

	optixCheck(
		optixProgramGroupCreate(
			optixSetup.getOptixContext(), 
			&pgDesc, 
			1, &
			pgOptions, 
			log,
			&sizeof_log,
			&sonelHitgroupPgs[SonelMapperRayTypes::DefaultRay]
		),
		"SonelMapper",
		"Failed to create closest hit and any hit programs."
	);
	if (sizeof_log > 1)
		PRINT(log);
}

/*! assembles the full pipeline of all programs */
void SonelMapper::createSonelPipeline() {
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : sonelRaygenPgs) {
		programGroups.push_back(pg);
	}
	for (auto pg : sonelHitgroupPgs) {
		programGroups.push_back(pg);
	}
	for (auto pg : sonelMissPgs) {
		programGroups.push_back(pg);
	}

	char log[2048];
	size_t sizeof_log = sizeof(log);
	PING;
	PRINT(programGroups.size());

	optixCheck(
		optixPipelineCreate(
			optixSetup.getOptixContext(),
			&sonelPipelineCompileOptions,
			&sonelPipelineLinkOptions,
			programGroups.data(),
			(int)programGroups.size(),
			log,
			&sizeof_log,
			&sonelPipeline
		),
		"SonelMapper",
		"Failed to create sonel pipeline."
	);

	if (sizeof_log > 1)
		PRINT(log);

	optixCheck(
		optixPipelineSetStackSize(
			/* [in] The pipeline to configure the stack size for */
			sonelPipeline,
			/* [in] The direct stack size requirement for
				direct callables invoked from IS or AH. */
			2 * 1024,
			/* [in] The direct stack size requirement for
				direct
				callables invoked from RG, MS, or CH.  */
			2 * 1024,
			/* [in] The continuation stack requirement. */
			2 * 1024,
			/* [in] The maximum depth of a traversable graph
				passed to trace. */
			1
		),
		"SonelMapper",
		"Failed to set stack size."
	);

	if (sizeof_log > 1)
		PRINT(log);
}


void SonelMapper::buildSonelRaygenRecords() {
	std::vector<RaygenRecord> raygenRecords;

	for (int i = 0; i < sonelRaygenPgs.size(); i++) {
		RaygenRecord rec;
		optixCheck(
			optixSbtRecordPackHeader(sonelRaygenPgs[i], &rec),
			"SonelMapper",
			"Failed to record sbt record header (raygen pgs)."
		);
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
	}

	sonelRaygenRecordsBuffer.allocAndUpload(raygenRecords);
	sonelSbt.raygenRecord = sonelRaygenRecordsBuffer.getCuDevicePointer();
}

void SonelMapper::buildSonelMissRecords() {
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < sonelMissPgs.size(); i++) {
		MissRecord rec;
		optixCheck(
			optixSbtRecordPackHeader(sonelMissPgs[i], &rec),
			"SonelMapper",
			"Failed to record sbt record header (miss pgs)."
		);
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}

	sonelMissRecordsBuffer.allocAndUpload(missRecords);
	sonelSbt.missRecordBase = sonelMissRecordsBuffer.getCuDevicePointer();

	sonelSbt.missRecordStrideInBytes = sizeof(MissRecord);
	sonelSbt.missRecordCount = (int)missRecords.size();
}

void SonelMapper::buildSonelHitgroupRecords() {
	const Model* model = cudaScene.getModel();
	int numObjects = (int)model->meshes.size();

	std::vector<HitgroupRecord> hitgroupRecords;
	for (int meshId = 0; meshId < numObjects; meshId++) {
		TriangleMesh* mesh = model->meshes[meshId];

		for (int rayId = 0; rayId < SonelMapperRayTypes::RaySize; rayId++) {
			HitgroupRecord rec;

			optixCheck(
				optixSbtRecordPackHeader(sonelHitgroupPgs[rayId], &rec),
				"SonelMapper",
				"Failed to record sbt record header (hitgroup pgs)."
			);

			cudaScene.fill(meshId, rec.data);

			hitgroupRecords.push_back(rec);
		}
	}

	sonelHitgroupRecordsBuffer.allocAndUpload(hitgroupRecords);
	sonelSbt.hitgroupRecordBase = sonelHitgroupRecordsBuffer.getCuDevicePointer();
	sonelSbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sonelSbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void SonelMapper::buildSonelSbt() {
	buildSonelRaygenRecords();
	buildSonelMissRecords();
	buildSonelHitgroupRecords();
}

/*! render one frame */
void SonelMapper::calculate() {	
	const Model* model = cudaScene.getModel();

	BoundingBox box = BoundingBox(model->bounds.lower, model->bounds.upper);
	uint32_t maxItems = 100;

	octTrees.resize(static_cast<uint64_t>(sonelMap.duration / sonelMap.timestep) + 1);
	sonelArrays.resize(static_cast<uint64_t>(sonelMap.duration / sonelMap.timestep) + 1);
	for (int i = 0; i < octTrees.size(); i++) {
		octTrees[i].init(box, maxItems);
	}

	for (uint32_t sourceIndex = 0; sourceIndex < sonelMap.soundSourceSize; sourceIndex++) {
        launchParams.soundSourceIndex = sourceIndex;
		SoundSource& soundSource = sonelMap.soundSources[sourceIndex];
		printf("[SonelMapper] Simulating sound source %d\n", sourceIndex);

		for (uint32_t fIndex = 0; fIndex < soundSource.frequencySize; fIndex++) {
			SoundFrequency& frequency = soundSource.frequencies[fIndex];
			printf("\tSimulating frequency (%d, %d)\n", fIndex, frequency.frequency);

			sonelMap.cudaUpload(sonelMapDevicePtr, sourceIndex, fIndex);
			launchParams.localFrequencyIndex = fIndex;
            launchParams.globalFrequencyIndex = sonelMap.getFrequencyIndex(frequency.frequency);
			launchOptix(frequency);
			downloadSonelDataForFrequency(fIndex, sourceIndex);
			sonelMap.cudaDestroy(sonelMapDevicePtr, sourceIndex, fIndex);
			cudaSyncCheck("SonelMapper", "Failed to sync.");
		}
	}
}

std::vector<std::vector<Sonel>>* SonelMapper::getSonelArrays() {
	return &sonelArrays;
}

void SonelMapper::launchOptix(SoundFrequency& frequency) {
	cudaMemcpy(launchParamsDevicePtr, &launchParams, sizeof(CudaSonelMapperParams), cudaMemcpyHostToDevice);

	optixCheck(
		optixLaunch(
			/*! pipeline we're launching launch: */
			sonelPipeline,
			optixSetup.getCudaStream(),

			/*! parameters and SBT */
			(CUdeviceptr)launchParamsDevicePtr,
			sizeof(CudaSonelMapperParams),
			&sonelSbt,

			/*! dimensions of the launch: */
			frequency.sonelAmount,
			frequency.decibelSize,
			1
		),
		"SonelMapper",
		"Failed to launch OptiX"
	);
}

void SonelMapper::downloadSonelDataForFrequency(uint32_t fIndex, uint32_t sourceIndex) {
	SoundFrequency& frequency = sonelMap.soundSources[sourceIndex].frequencies[fIndex];
	
	sonelMap.cudaDownload(sonelMapDevicePtr, sourceIndex, fIndex);
	Sonel* sonels = frequency.sonels;

	// Go over all rays
	uint64_t sonelAmount = 0;
	for (int i = 0; i < frequency.sonelAmount; i++) {
		// Go over each bounce in the ray
		for (int j = 0; j < frequency.sonelMaxDepth; j++) {
			Sonel& sonel = sonels[i * frequency.sonelMaxDepth + j];
			sonel.frequency = frequency.frequency;

			// The energies of a sonel is 0 the ray is absorbed and done.
			if (sonel.energy < 0.00001f) {
				break;
			}

			uint64_t timeIndex = static_cast<uint64_t>(sonel.time / sonelMap.timestep);

			if (timeIndex < octTrees.size()) {
				sonelAmount++;
				octTrees[timeIndex].insert(&sonel, sonel.position);
				sonelArrays[timeIndex].push_back(sonel);
			}
		}
	}

	printf("[SonelMapper] Sonels added %d\n", sonelAmount);
}

const SonelMapData &SonelMapper::getSonelMapData() const {
    return sonelMap;
}
