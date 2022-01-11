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

#include "SonelMapper.h"

#include "LaunchParams.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#include "Model.h"
#include "OctTree.h"

extern "C" char embedded_ptx_code[];

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

	TriangleMeshSBTData data;
};

/*! constructor - performs all setup, including initializing
	optix, creates module, pipeline, programs, SBT, etc. */
SonelMapper::SonelMapper(
	const Model* model, 
	const QuadLight& light, 
	SoundSource soundSource, 
	float echogramDuration, 
	float soundSpeed, 
	float earSize
): model(model) {
	initOptix();

	launchParams.light.origin = light.origin;
	launchParams.light.du = light.du;
	launchParams.light.dv = light.dv;
	launchParams.light.power = light.power;
	launchParams.sonelMap.soundSource = soundSource;
	launchParams.sonelMap.echogramDuration = echogramDuration;
	launchParams.sonelMap.soundSpeed = soundSpeed;
	launchParams.sonelMap.earSize = earSize;
	launchParams.sonelMap.sonelMaxDepth = 8;
	launchParams.sonelMap.sonelAmount = 100000;
	launchParams.sonelMap.sonelBufferSize = launchParams.sonelMap.sonelMaxDepth * launchParams.sonelMap.sonelAmount;
	launchParams.octTree = nullptr;
	initSonelBuffer();
	

	std::cout << "#osc: creating optix context ..." << std::endl;
	createContext();

	std::cout << "#osc: setting up module ..." << std::endl;
	createRenderModule();
	createSonelModule();

	std::cout << "#osc: creating raygen programs ..." << std::endl;
	createRenderRaygenPrograms();
	createSonelRaygenPrograms();
	std::cout << "#osc: creating miss programs ..." << std::endl;
	createRenderMissPrograms();
	createSonelMissPrograms();
	std::cout << "#osc: creating hitgroup programs ..." << std::endl;
	createRenderHitgroupPrograms();
	createSonelHitgroupPrograms();

	launchParams.traversable = buildAccel();

	std::cout << "#osc: setting up optix pipeline ..." << std::endl;
	createRenderPipeline();
	createSonelPipeline();

	createTextures();

	std::cout << "#osc: building SBT ..." << std::endl;
	buildRenderSbt();
	buildSonelSbt();

	launchParamsBuffer.alloc(sizeof(launchParams));
	std::cout << "#osc: context, module, pipeline, etc, all set up ..."
		<< std::endl;

	std::cout << GDT_TERMINAL_GREEN;
	std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
	std::cout << GDT_TERMINAL_DEFAULT;
}

void SonelMapper::createTextures() {
	int numTextures = (int)model->textures.size();

	textureArrays.resize(numTextures);
	textureObjects.resize(numTextures);

	for (int textureID = 0; textureID < numTextures; textureID++) {
		auto texture = model->textures[textureID];

		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc;
		int32_t width = texture->resolution.x;
		int32_t height = texture->resolution.y;
		int32_t numComponents = 4;
		int32_t pitch = width * numComponents * sizeof(uint8_t);
		channel_desc = cudaCreateChannelDesc<uchar4>();

		cudaArray_t& pixelArray = textureArrays[textureID];
		CUDA_CHECK(MallocArray(&pixelArray, &channel_desc, width, height));

		CUDA_CHECK(Memcpy2DToArray(pixelArray,
			/* offset */ 0, 0, texture->pixel, pitch, pitch,
			height, cudaMemcpyHostToDevice));

		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = pixelArray;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 99;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.borderColor[0] = 1.0f;
		tex_desc.sRGB = 0;

		// Create texture object
		cudaTextureObject_t cuda_tex = 0;
		CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
		textureObjects[textureID] = cuda_tex;
	}
}

OptixTraversableHandle SonelMapper::buildAccel() {
	const int numMeshes = (int)model->meshes.size();
	vertexBuffer.resize(numMeshes);
	normalBuffer.resize(numMeshes);
	texcoordBuffer.resize(numMeshes);
	indexBuffer.resize(numMeshes);

	OptixTraversableHandle asHandle{ 0 };

	// ==================================================================
	// triangle inputs
	// ==================================================================
	std::vector<OptixBuildInput> triangleInput(numMeshes);
	std::vector<CUdeviceptr> d_vertices(numMeshes);
	std::vector<CUdeviceptr> d_indices(numMeshes);
	std::vector<uint32_t> triangleInputFlags(numMeshes);

	for (int meshID = 0; meshID < numMeshes; meshID++) {
		// upload the model to the device: the builder
		TriangleMesh& mesh = *model->meshes[meshID];
		vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
		indexBuffer[meshID].alloc_and_upload(mesh.index);
		if (!mesh.normal.empty())
			normalBuffer[meshID].alloc_and_upload(mesh.normal);
		if (!mesh.texcoord.empty())
			texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

		triangleInput[meshID] = {};
		triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// create local variables, because we need a *pointer* to the
		// device pointers
		d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
		d_indices[meshID] = indexBuffer[meshID].d_pointer();

		triangleInput[meshID].triangleArray.vertexFormat =
			OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
		triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
		triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

		triangleInput[meshID].triangleArray.indexFormat =
			OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
		triangleInput[meshID].triangleArray.numIndexTriplets =
			(int)mesh.index.size();
		triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

		triangleInputFlags[meshID] = 0;

		// in this example we have one SBT entry, and no per-primitive
		// materials:
		triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
		triangleInput[meshID].triangleArray.numSbtRecords = 1;
		triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}
	// ==================================================================
	// BLAS setup
	// ==================================================================

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags =
		OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions,
		triangleInput.data(),
		(int)numMeshes, // num_build_inputs
		&blasBufferSizes));

	// ==================================================================
	// prepare compaction
	// ==================================================================

	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	// ==================================================================
	// execute build (main stage)
	// ==================================================================

	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(
		optixContext,
		/* stream */ 0, &accelOptions, triangleInput.data(), (int)numMeshes,
		tempBuffer.d_pointer(), tempBuffer.sizeInBytes,

		outputBuffer.d_pointer(), outputBuffer.sizeInBytes,

		&asHandle,

		&emitDesc, 1));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// perform compaction
	// ==================================================================
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	asBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(optixContext,
		/*stream:*/ 0, asHandle, asBuffer.d_pointer(),
		asBuffer.sizeInBytes, &asHandle));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// aaaaaand .... clean up
	// ==================================================================
	outputBuffer.free(); // << the UNcompacted, temporary output buffer
	tempBuffer.free();
	compactedSizeBuffer.free();

	return asHandle;
}

/*! helper function that initializes optix and checks for errors */
void SonelMapper::initOptix() {
	std::cout << "#osc: initializing optix..." << std::endl;

	// -------------------------------------------------------
	// check for available optix7 capable devices
	// -------------------------------------------------------
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw std::runtime_error("#osc: no CUDA capable devices found!");
	std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

	// -------------------------------------------------------
	// initialize optix
	// -------------------------------------------------------
	OPTIX_CHECK(optixInit());
	std::cout << GDT_TERMINAL_GREEN
		<< "#osc: successfully initialized optix... yay!"
		<< GDT_TERMINAL_DEFAULT << std::endl;
}

static void context_log_cb(unsigned int level, const char* tag,
	const char* message, void*) {
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

/*! creates and configures a optix device context (in this simple
	example, only for the primary GPU device) */
void SonelMapper::createContext() {
	// for this sample, do everything on one device
	const int deviceID = 0;
	CUDA_CHECK(SetDevice(deviceID));
	CUDA_CHECK(StreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb,
		nullptr, 4));
}

/*! creates the module that contains all the programs we are going
	to use. in this simple example, we use a single module from a
	single .cu file, using a single embedded ptx string */
void SonelMapper::createRenderModule() {
	renderModuleCompileOptions.maxRegisterCount = 50;
	renderModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	renderModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	renderPipelineCompileOptions = {};
	renderPipelineCompileOptions.traversableGraphFlags =
		OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	renderPipelineCompileOptions.usesMotionBlur = false;
	renderPipelineCompileOptions.numPayloadValues = 2;
	renderPipelineCompileOptions.numAttributeValues = 2;
	renderPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	renderPipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";

	renderPipelineLinkOptions.maxTraceDepth = 2;

	const std::string ptxCode = embedded_ptx_code;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixModuleCreateFromPTX(
		optixContext, &renderModuleCompileOptions, &renderPipelineCompileOptions,
		ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &renderModule));
	if (sizeof_log > 1)
		PRINT(log);
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
	sonelPipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";

	sonelPipelineLinkOptions.maxTraceDepth = 8;

	const std::string ptxCode = embedded_ptx_code;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(
		optixModuleCreateFromPTX(
			optixContext, \
			&sonelModuleCompileOptions, 
			&sonelPipelineCompileOptions,
			ptxCode.c_str(), 
			ptxCode.size(), 
			log, 
			&sizeof_log, 
			&sonelModule
		)
	);

	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the raygen program(s) we are going to use */
void SonelMapper::createRenderRaygenPrograms() {
	// we do a single ray gen program in this example:
	renderRaygenPgs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = renderModule;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(
		optixProgramGroupCreate(
			optixContext, 
			&pgDesc, 
			1, 
			&pgOptions, 
			log,
			&sizeof_log, 
			&renderRaygenPgs[0]
		)
	);
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the raygen program(s) we are going to use */
void SonelMapper::createSonelRaygenPrograms() {
	// we do a single ray gen program in this example:
	sonelRaygenPgs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = sonelModule;
	pgDesc.raygen.entryFunctionName = "__raygen__generateSonelMap";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(
		optixProgramGroupCreate(
			optixContext,
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&sonelRaygenPgs[0]
		)
	);
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the miss program(s) we are going to use */
void SonelMapper::createRenderMissPrograms() {
	// we do a single ray gen program in this example:
	renderMissPgs.resize(RAY_TYPE_COUNT);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = renderModule;

	// ------------------------------------------------------------------
	// radiance rays
	// ------------------------------------------------------------------
	pgDesc.miss.entryFunctionName = "__miss__radiance";

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
		&sizeof_log,
		&renderMissPgs[RADIANCE_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);

	// ------------------------------------------------------------------
	// shadow rays
	// ------------------------------------------------------------------
	pgDesc.miss.entryFunctionName = "__miss__shadow";

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
		&sizeof_log, &renderMissPgs[SHADOW_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the miss program(s) we are going to use */
void SonelMapper::createSonelMissPrograms() {
	// we do a single ray gen program in this example:
	sonelMissPgs.resize(RAY_TYPE_COUNT);

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

	OPTIX_CHECK(
		optixProgramGroupCreate(
			optixContext, 
			&pgDesc, 
			1, 
			&pgOptions, 
			log,
			&sizeof_log,
			&sonelMissPgs[RADIANCE_RAY_TYPE]
		)
	);

	if (sizeof_log > 1)
		PRINT(log);

	// ------------------------------------------------------------------
	// shadow rays
	// ------------------------------------------------------------------
	pgDesc.miss.entryFunctionName = "__miss__sonelShadow";

	OPTIX_CHECK(
		optixProgramGroupCreate(
			optixContext, 
			&pgDesc, 
			1, 
			&pgOptions, 
			log,
			&sizeof_log, 
			&sonelMissPgs[SHADOW_RAY_TYPE]
		)
	);

	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void SonelMapper::createRenderHitgroupPrograms() {
	// for this simple example, we set up a single hit group
	renderHitgroupPgs.resize(RAY_TYPE_COUNT);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = renderModule;
	pgDesc.hitgroup.moduleAH = renderModule;

	// -------------------------------------------------------
	// radiance rays
	// -------------------------------------------------------
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
		&sizeof_log,
		&renderHitgroupPgs[RADIANCE_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);

	// -------------------------------------------------------
	// shadow rays: technically we don't need this hit group,
	// since we just use the miss shader to check if we were not
	// in shadow
	// -------------------------------------------------------
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
		&sizeof_log,
		&renderHitgroupPgs[SHADOW_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void SonelMapper::createSonelHitgroupPrograms() {
	// for this simple example, we set up a single hit group
	sonelHitgroupPgs.resize(RAY_TYPE_COUNT);

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

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
		&sizeof_log,
		&sonelHitgroupPgs[RADIANCE_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);

	// -------------------------------------------------------
	// shadow rays: technically we don't need this hit group,
	// since we just use the miss shader to check if we were not
	// in shadow
	// -------------------------------------------------------
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__sonelShadow";
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__sonelShadow";

	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
		&sizeof_log,
		&sonelHitgroupPgs[SHADOW_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);
}

/*! assembles the full pipeline of all programs */
void SonelMapper::createRenderPipeline() {
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : renderRaygenPgs) {
		programGroups.push_back(pg);
	}
	for (auto pg : renderHitgroupPgs) {
		programGroups.push_back(pg);
	}
	for (auto pg : renderMissPgs) {
		programGroups.push_back(pg);
	}

	char log[2048];
	size_t sizeof_log = sizeof(log);
	PING;
	PRINT(programGroups.size());

	OPTIX_CHECK(
		optixPipelineCreate(
			optixContext, 
			&renderPipelineCompileOptions,
			&renderPipelineLinkOptions, 
			programGroups.data(),
			(int)programGroups.size(), 
			log, 
			&sizeof_log,
			&renderPipeline
		)
	);

	if (sizeof_log > 1)
		PRINT(log);

	OPTIX_CHECK(
		optixPipelineSetStackSize(
			/* [in] The pipeline to configure the stack size for */
			renderPipeline,
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
		)
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

	OPTIX_CHECK(
		optixPipelineCreate(
			optixContext,
			&sonelPipelineCompileOptions,
			&sonelPipelineLinkOptions,
			programGroups.data(),
			(int)programGroups.size(),
			log,
			&sizeof_log,
			&sonelPipeline
		)
	);

	if (sizeof_log > 1)
		PRINT(log);

	OPTIX_CHECK(
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
		)
	);

	if (sizeof_log > 1)
		PRINT(log);
}

void SonelMapper::buildRenderRaygenRecords() {
	std::vector<RaygenRecord> raygenRecords;

	for (int i = 0; i < renderRaygenPgs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(renderRaygenPgs[i], &rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
	}

	renderRaygenRecordsBuffer.alloc_and_upload(raygenRecords);
	renderSbt.raygenRecord = renderRaygenRecordsBuffer.d_pointer();
}

void SonelMapper::buildSonelRaygenRecords() {
	std::vector<RaygenRecord> raygenRecords;

	for (int i = 0; i < sonelRaygenPgs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(sonelRaygenPgs[i], &rec));
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
	}

	sonelRaygenRecordsBuffer.alloc_and_upload(raygenRecords);
	sonelSbt.raygenRecord = sonelRaygenRecordsBuffer.d_pointer();
}

void SonelMapper::buildRenderMissRecords() {
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < renderMissPgs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(renderMissPgs[i], &rec));
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}

	renderMissRecordsBuffer.alloc_and_upload(missRecords);
	renderSbt.missRecordBase = renderMissRecordsBuffer.d_pointer();

	renderSbt.missRecordStrideInBytes = sizeof(MissRecord);
	renderSbt.missRecordCount = (int)missRecords.size();
}

void SonelMapper::buildSonelMissRecords() {
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < sonelMissPgs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(sonelMissPgs[i], &rec));
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}

	sonelMissRecordsBuffer.alloc_and_upload(missRecords);
	sonelSbt.missRecordBase = sonelMissRecordsBuffer.d_pointer();

	sonelSbt.missRecordStrideInBytes = sizeof(MissRecord);
	sonelSbt.missRecordCount = (int)missRecords.size();
}

void SonelMapper::buildRenderHitgroupRecords() {
	int numObjects = (int)model->meshes.size();

	std::vector<HitgroupRecord> hitgroupRecords;
	for (int meshId = 0; meshId < numObjects; meshId++) {
		TriangleMesh* mesh = model->meshes[meshId];

		for (int rayId = 0; rayId < RAY_TYPE_COUNT; rayId++) {
			HitgroupRecord rec;

			OPTIX_CHECK(optixSbtRecordPackHeader(renderHitgroupPgs[rayId], &rec));

			// Load color data
			rec.data.color = mesh->diffuse;
			if (mesh->diffuseTextureId >= 0) {
				rec.data.hasTexture = true;
				rec.data.texture = textureObjects[mesh->diffuseTextureId];
			}
			else {
				rec.data.hasTexture = false;
			}

			// Load vector data
			rec.data.index = (vec3i*)indexBuffer[meshId].d_pointer();
			rec.data.vertex = (vec3f*)vertexBuffer[meshId].d_pointer();
			rec.data.normal = (vec3f*)normalBuffer[meshId].d_pointer();
			rec.data.texcoord = (vec2f*)texcoordBuffer[meshId].d_pointer();

			hitgroupRecords.push_back(rec);
		}
	}

	renderHitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	renderSbt.hitgroupRecordBase = renderHitgroupRecordsBuffer.d_pointer();
	renderSbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	renderSbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void SonelMapper::buildSonelHitgroupRecords() {
	int numObjects = (int)model->meshes.size();

	std::vector<HitgroupRecord> hitgroupRecords;
	for (int meshId = 0; meshId < numObjects; meshId++) {
		TriangleMesh* mesh = model->meshes[meshId];

		for (int rayId = 0; rayId < RAY_TYPE_COUNT; rayId++) {
			HitgroupRecord rec;

			OPTIX_CHECK(optixSbtRecordPackHeader(sonelHitgroupPgs[rayId], &rec));

			// Load color data
			rec.data.color = mesh->diffuse;
			if (mesh->diffuseTextureId >= 0) {
				rec.data.hasTexture = true;
				rec.data.texture = textureObjects[mesh->diffuseTextureId];
			}
			else {
				rec.data.hasTexture = false;
			}

			// Load vector data
			rec.data.index = (vec3i*)indexBuffer[meshId].d_pointer();
			rec.data.vertex = (vec3f*)vertexBuffer[meshId].d_pointer();
			rec.data.normal = (vec3f*)normalBuffer[meshId].d_pointer();
			rec.data.texcoord = (vec2f*)texcoordBuffer[meshId].d_pointer();

			hitgroupRecords.push_back(rec);
		}
	}

	sonelHitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sonelSbt.hitgroupRecordBase = sonelHitgroupRecordsBuffer.d_pointer();
	sonelSbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sonelSbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

/*! constructs the shader binding table */
void SonelMapper::buildRenderSbt() {
	buildRenderRaygenRecords();
	buildRenderMissRecords();
	buildRenderHitgroupRecords();
}

void SonelMapper::buildSonelSbt() {
	buildSonelRaygenRecords();
	buildSonelMissRecords();
	buildSonelHitgroupRecords();
}

/*! render one frame */
void SonelMapper::render() {
	// sanity check: make sure we launch only after first resize is
	// already done:
	if (launchParams.frame.size.x == 0) {
		return;
	}

	launchParamsBuffer.upload(&launchParams, 1);

	if (!hasCalculatedSonelMap) {
		OPTIX_CHECK(
			optixLaunch(
				/*! pipeline we're launching launch: */
				sonelPipeline,
				stream,

				/*! parameters and SBT */
				launchParamsBuffer.d_pointer(),
				launchParamsBuffer.sizeInBytes,
				&sonelSbt,

				/*! dimensions of the launch: */
				launchParams.sonelMap.sonelAmount,
				1,
				1
			)
		);
		
		CUDA_SYNC_CHECK();
		hasCalculatedSonelMap = true;

		std::vector<Sonel> sonelMap;
		sonelMap.resize(launchParams.sonelMap.sonelBufferSize);
		downloadSonelMap(sonelMap.data());

		this->sonelMap = new SonelMap(sonelMap.data(), launchParams.sonelMap.sonelAmount, launchParams.sonelMap.sonelMaxDepth, launchParams.sonelMap.echogramDuration, launchParams.sonelMap.soundSpeed);
		sonelMapIndex = 0;
	}

	uploadSonelMapSnapshot(sonelMapIndex);
	sonelMapIndex++;

	if (sonelMapIndex == sonelMap->getTimestepAmount() - 1) {
		sonelMapIndex = 0;
	}

	launchParamsBuffer.upload(&launchParams, 1);

	OPTIX_CHECK(
		optixLaunch(
			/*! pipeline we're launching launch: */
			renderPipeline, 
			stream,

			/*! parameters and SBT */
			launchParamsBuffer.d_pointer(),
			launchParamsBuffer.sizeInBytes, 
			&renderSbt,

			/*! dimensions of the launch: */
			launchParams.frame.size.x,
			launchParams.frame.size.y,
			1
		)
	);

	// sync - make sure the frame is rendered before we download and
	// display (obviously, for a high-performance application you
	// want to use streams and double-buffering, but for this simple
	// example, this will have to do)
	CUDA_SYNC_CHECK();
}

/*! set camera to render with */
void SonelMapper::setCamera(const Camera& camera) {
	lastSetCamera = camera;
	launchParams.camera.position = camera.from;
	launchParams.camera.direction = normalize(camera.at - camera.from);

	const float cosFovy = 0.66f;
	const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);

	launchParams.camera.horizontal = cosFovy * aspect * normalize(cross(launchParams.camera.direction, camera.up));
	launchParams.camera.vertical = cosFovy * normalize(cross(launchParams.camera.horizontal, launchParams.camera.direction));
}

/*! resize frame buffer to given resolution */
void SonelMapper::resize(const vec2i& newSize) {
	// if window minimized
	if (newSize.x == 0 || newSize.y == 0)
		return;

	// resize our cuda frame buffer
	uint64_t bufferSize = (uint64_t)newSize.x *  (uint64_t)newSize.y * sizeof(uint32_t);
	colorBuffer.resize(bufferSize);

	// update the launch parameters that we'll pass to the optix
	// launch:
	launchParams.frame.size = newSize;
	launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();

	// and re-set the camera, since aspect may have changed
	setCamera(lastSetCamera);
}

void SonelMapper::initSonelBuffer() {
	sonelMapBuffer.resize(launchParams.sonelMap.sonelBufferSize * sizeof(Sonel));
	launchParams.sonelMap.sonelBuffer = (Sonel*)sonelMapBuffer.d_pointer();
}

void SonelMapper::downloadSonelMap(Sonel sonels[]) {
	sonelMapBuffer.download(sonels, launchParams.sonelMap.sonelBufferSize);
}

void SonelMapper::uploadSonelMapSnapshot(int index) {
	printf("Uploading index %d\n", index);
	std::vector<Sonel> sonels;
	
	sonelMap->getTimestep(index, sonels);
	if (sonels.size() == 0) {
		sonels.push_back({
			gdt::vec3f(),
			gdt::vec3f(),
			0,
			0,
			0
			});
	}

	BoundingBox box(model->bounds.lower, model->bounds.upper);
	uint32_t maxItems = 20;
	OctTree<Sonel> octTree(box, maxItems);

	for (int i = 0; i < sonels.size(); i++) {
		octTree.insert(&(sonels[i]), sonels[i].position);
	}

	// OctTree<Sonel>* deviceOctTree = octTree.upload();

	sonelMapBuffer.resize(sonels.size() * sizeof(Sonel));
	sonelMapBuffer.upload(sonels.data(), sonels.size());
	launchParams.sonelMap.sonelBuffer = (Sonel*)sonelMapBuffer.d_pointer();
	launchParams.sonelMap.sonelBufferSize = sonels.size();
	// launchParams.octTree = deviceOctTree;
}

/*! download the rendered color buffer */
void SonelMapper::downloadPixels(uint32_t h_pixels[]) {
	uint64_t bufferSize = (uint64_t)launchParams.frame.size.x * (uint64_t)launchParams.frame.size.y;

	colorBuffer.download(h_pixels, bufferSize);
}