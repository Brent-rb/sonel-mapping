//
// Created by brent on 27/01/2022.
//

#ifndef SONEL_MAPPING_SMOPTIXPROGRAM_H
#define SONEL_MAPPING_SMOPTIXPROGRAM_H

#include <functional>

#include "OptixSetup.h"
#include "OptixScene.h"


struct EmptyRecord {

};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SmRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	T data;
};

template<typename X, typename U, typename V, typename W>
class SmOptixProgram {
public:
	virtual void execute() = 0;

protected:
	SmOptixProgram(
		const std::string optixCode,
		const OptixSetup& optixSetup,
		const OptixScene& optixScene,
		const uint32_t raygenProgramSize = 0,
		const uint32_t missProgramSize = 0,
		const uint32_t hitProgramSize = 0
	): optixCode(optixCode), optixSetup(optixSetup), optixScene(optixScene),
	   raygenProgramSize(raygenProgramSize), missProgramSize(missProgramSize), hitgroupProgramSize(hitProgramSize),
	   pipeline(nullptr), module(nullptr) {

		X* launchParamsDevicePtr;
		cudaMalloc(&launchParamsDevicePtr, sizeof(X));
		launchParamsPtr = reinterpret_cast<CUdeviceptr>(launchParamsDevicePtr);
	}

	~SmOptixProgram() {
		cudaFree(reinterpret_cast<void*>(launchParamsPtr));
		raygenRecordsBuffer.tryFree();
		missRecordsBuffer.tryFree();
		hitgroupRecordsBuffer.tryFree();
	}

	void init() {
		createOptixModule();
		createRaygenPrograms();
		createMissPrograms();
		createHitPrograms();
		createPipeline();
		createSbt();
	}

	virtual void configureModuleCompileOptions(OptixModuleCompileOptions& compileOptions) {};
	virtual void configurePipelineCompileOptions(OptixPipelineCompileOptions& pipelineOptions) {};
	virtual void configurePipelineLinkOptions(OptixPipelineLinkOptions& pipelineLinkOptions) {};

	virtual const char* getLaunchParamsName() = 0;
	virtual void configureRaygenProgram(uint32_t programIndex, OptixProgramGroupOptions& options, OptixProgramGroupDesc& desc) = 0;
	virtual void configureMissProgram(uint32_t programIndex, OptixProgramGroupOptions& options, OptixProgramGroupDesc& desc) = 0;
	virtual void configureHitProgram(uint32_t programIndex, OptixProgramGroupOptions& options, OptixProgramGroupDesc& desc) = 0;

	void createSbt() {
		createRaygenRecords();
		createMissRecords();
		createHitRecords();
	}

	void createRaygenRecords() {
		std::vector<SmRecord<U>> raygenRecords;

		for (uint32_t i = 0; i < raygenProgramSize; i++) {
			OptixProgramGroup& raygenPg = raygenPgs[i];
			SmRecord<U> tempRecord = {};

			optixCheck(
				optixSbtRecordPackHeader(raygenPg, &tempRecord),
				SM_OPTIX_PROGRAM_PREFIX,
				"Failed to record sbt record header (raygen pgs)."
			);

			addRaygenRecords(i, tempRecord);
			raygenRecords.push_back(tempRecord);
		}

		raygenRecordsBuffer.allocAndUpload(raygenRecords);
		sbt.raygenRecord = raygenRecordsBuffer.getCuDevicePointer();
	}

	virtual void addRaygenRecords(uint32_t programIndex, SmRecord<U>& raygenRecord) {

	};

	void createMissRecords() {
		std::vector<SmRecord<V>> missRecords;

		for (uint32_t i = 0; i < missProgramSize; i++) {
			OptixProgramGroup& missPg = missPgs[i];
			SmRecord<V> tempRecord = {};

			optixCheck(
				optixSbtRecordPackHeader(missPg, &tempRecord),
				SM_OPTIX_PROGRAM_PREFIX,
				"Failed to record sbt record header (miss pgs)."
			);

			addMissRecords(i, tempRecord);
			missRecords.push_back(tempRecord);
		}

		missRecordsBuffer.allocAndUpload(missRecords);
		sbt.missRecordBase = missRecordsBuffer.getCuDevicePointer();
		sbt.missRecordStrideInBytes = sizeof(SmRecord<U>);
		sbt.missRecordCount = static_cast<unsigned int>(missRecords.size());
	}

	virtual void addMissRecords(uint32_t programIndex, SmRecord<V>& missRecords) {

	};

	void createHitRecords() {
		std::vector<SmRecord<W>> hitRecords;

		addHitRecords(hitRecords);

		hitgroupRecordsBuffer.allocAndUpload(hitRecords);
		sbt.hitgroupRecordBase = hitgroupRecordsBuffer.getCuDevicePointer();
		sbt.hitgroupRecordStrideInBytes = sizeof(SmRecord<W>);
		sbt.hitgroupRecordCount = static_cast<unsigned int>(hitRecords.size());
	}

	virtual void addHitRecords(std::vector<SmRecord<W>>& hitRecords) = 0;

	void launchOptix(uint32_t x, uint32_t y, uint32_t z) {
		cudaMemcpy(reinterpret_cast<void*>(launchParamsPtr), &launchParams, sizeof(X), cudaMemcpyHostToDevice);

		optixCheck(
			optixLaunch(
				/*! pipeline we're launching launch: */
				pipeline,
				optixSetup.getCudaStream(),

				/*! parameters and SBT */
				launchParamsPtr,
				sizeof(X),
				&sbt,

				/*! dimensions of the launch: */
				x,
				y,
				z
			),
			"SonelMapper",
			"Failed to launch OptiX"
		);
	}

private:
	void createOptixModule() {
		// Apply default config
		moduleCompileOptions.maxRegisterCount = 50;
		moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
		// Allow child to config as well
		configureModuleCompileOptions(moduleCompileOptions);

		// Apply default config
		pipelineCompileOptions = {};
		pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipelineCompileOptions.usesMotionBlur = false;
		pipelineCompileOptions.numPayloadValues = 2;
		pipelineCompileOptions.numAttributeValues = 2;
		pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipelineCompileOptions.pipelineLaunchParamsVariableName = getLaunchParamsName();
		// Allow child to config as well
		configurePipelineCompileOptions(pipelineCompileOptions);

		// Apply default config
		pipelineLinkOptions.maxTraceDepth = 8;
		configurePipelineLinkOptions(pipelineLinkOptions);

		const std::string ptxCode = optixCode;

		char log[2048];
		size_t sizeof_log = sizeof(log);
		optixCheck(
				optixModuleCreateFromPTX(
						optixSetup.getOptixContext(),
						&moduleCompileOptions,
						&pipelineCompileOptions,
						ptxCode.c_str(),
						ptxCode.size(),
						log,
						&sizeof_log,
						&module
				),
				SM_OPTIX_PROGRAM_PREFIX,
				"Failed to create optix module."
		);

		if (sizeof_log > 1)
			PRINT(log)
	}

	void createPrograms(
		std::vector<OptixProgramGroup>& group,
		uint32_t size,
		std::function<void(SmOptixProgram*, OptixProgramGroupDesc&)> configureDesc,
		std::function<void(SmOptixProgram*, uint32_t, OptixProgramGroupOptions&, OptixProgramGroupDesc&)> configureFunction
	) {
		std::vector<OptixProgramGroupOptions> options;
		std::vector<OptixProgramGroupDesc> descs;
		group.resize(size);
		options.resize(size);
		descs.resize(size);

		for (uint32_t i = 0; i < size; i++) {
			OptixProgramGroupOptions& pgOptions = options[i];
			OptixProgramGroupDesc& pgDesc = descs[i];

			configureDesc(this, pgDesc);
			configureFunction(this, i, pgOptions, pgDesc);
		}

		char log[2048];
		size_t sizeof_log = sizeof(log);

		optixCheck(
			optixProgramGroupCreate(
				optixSetup.getOptixContext(),
				descs.data(),
				size,
				options.data(),
				log,
				&sizeof_log,
				group.data()
			),
			SM_OPTIX_PROGRAM_PREFIX,
			"Failed to programs."
		);

		if (sizeof_log > 1) {
			PRINT(log)
		}
	}

	void createRaygenPrograms() {
		createPrograms(raygenPgs, raygenProgramSize, &SmOptixProgram::configureRaygenDesc, &SmOptixProgram::configureRaygenProgram);
	}

	void configureRaygenDesc(OptixProgramGroupDesc& desc) {
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		desc.raygen.module = module;
	}

	void createMissPrograms() {
		createPrograms(missPgs, missProgramSize, &SmOptixProgram::configureMissDesc, &SmOptixProgram::configureMissProgram);
	}

	void configureMissDesc(OptixProgramGroupDesc& desc) {
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		desc.miss.module = module;
	}

	void createHitPrograms() {
		createPrograms(hitgroupPgs, hitgroupProgramSize, &SmOptixProgram::configureHitDesc, &SmOptixProgram::configureHitProgram);
	}

	void configureHitDesc(OptixProgramGroupDesc& desc) {
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		desc.hitgroup.moduleCH = module;
		if (hitAhEnabled)
			desc.hitgroup.moduleAH = module;
		if (hitIsEnabled)
			desc.hitgroup.moduleIS = module;
	}

	void createPipeline() {
		std::vector<OptixProgramGroup> programGroups;
		for (auto pg : raygenPgs) {
			programGroups.push_back(pg);
		}
		for (auto pg : hitgroupPgs) {
			programGroups.push_back(pg);
		}
		for (auto pg : missPgs) {
			programGroups.push_back(pg);
		}

		char log[2048];
		size_t sizeof_log = sizeof(log);
		PING
		PRINT(programGroups.size())

		optixCheck(
		optixPipelineCreate(
				optixSetup.getOptixContext(),
				&pipelineCompileOptions,
				&pipelineLinkOptions,
				programGroups.data(),
				(int)programGroups.size(),
				log,
				&sizeof_log,
				&pipeline
			),
			SM_OPTIX_PROGRAM_PREFIX,
			"Failed to create sonel pipeline."
		);

		if (sizeof_log > 1)
			PRINT(log)

		optixCheck(
		optixPipelineSetStackSize(
				/* [in] The pipeline to configure the stack size for */
				pipeline,
				/* [in] The direct stack size requirement for
					direct callables invoked from IS or AH. */
				stackSizeTraversal,
				/* [in] The direct stack size requirement for
					direct
					callables invoked from RG, MS, or CH.  */
				stackSizeState,
				/* [in] The continuation stack requirement. */
				continuationStackSize,
				/* [in] The maximum depth of a traversable graph
					passed to trace. */
				maxTraversableGraphDepth
			),
			SM_OPTIX_PROGRAM_PREFIX,
			"Failed to set stack size."
		);

		if (sizeof_log > 1) {
			PRINT(log)
		}
	}

protected:
	X launchParams;

	const std::string optixCode;
	const OptixSetup& optixSetup;
	const OptixScene& optixScene;

	std::vector<OptixProgramGroup> raygenPgs;
	std::vector<OptixProgramGroup> missPgs;
	std::vector<OptixProgramGroup> hitgroupPgs;
	const uint32_t raygenProgramSize;
	const uint32_t missProgramSize;
	const uint32_t hitgroupProgramSize;

	uint32_t maxTraversableGraphDepth = 1;
	uint32_t stackSizeTraversal = 2 * 1024;
	uint32_t stackSizeState = 2 * 1024;
	uint32_t continuationStackSize = 2 * 1024;

	bool hitIsEnabled = false;
	bool hitAhEnabled = false;

private:
	// Pipeline
	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};

	// Module
	OptixModule module;
	OptixModuleCompileOptions moduleCompileOptions = {};

	// Programs
	CudaBuffer raygenRecordsBuffer;
	CudaBuffer missRecordsBuffer;
	CudaBuffer hitgroupRecordsBuffer;

	OptixShaderBindingTable sbt = {};
	CUdeviceptr launchParamsPtr;

	const char* SM_OPTIX_PROGRAM_PREFIX = "SmOptixProgram";
};


#endif //SONEL_MAPPING_SMOPTIXPROGRAM_H
