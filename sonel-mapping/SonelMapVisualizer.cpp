#include "SonelMapVisualizer.h"

#include "LaunchParams.h"
#include "OctTree.h"
#include "TriangleMeshSbtData.h"
#include "CudaHelper.h"

extern "C" char embedded_visualizer_code[];

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

/*! constructor - performs all setup, including initializing
	optix, creates module, pipeline, programs, SBT, etc. */
SonelMapVisualizer::SonelMapVisualizer(
	const OptixSetup& optixSetup,
	const OptixScene& cudaScene
): optixSetup(optixSetup), cudaScene(cudaScene), sonelMap(nullptr) {
	launchParams.octTree = nullptr;

	std::cout << "#osc: setting up module ..." << std::endl;
	createRenderModule();

	std::cout << "#osc: creating raygen programs ..." << std::endl;
	createRenderRaygenPrograms();

	std::cout << "#osc: creating miss programs ..." << std::endl;
	createRenderMissPrograms();

	std::cout << "#osc: creating hitgroup programs ..." << std::endl;
	createRenderHitgroupPrograms();

	launchParams.traversable = cudaScene.getTraversableHandle();

	std::cout << "#osc: setting up optix pipeline ..." << std::endl;
	createRenderPipeline();


	std::cout << "#osc: building SBT ..." << std::endl;
	buildRenderSbt();

	launchParamsBuffer.alloc(sizeof(launchParams));
	std::cout << "#osc: context, module, pipeline, etc, all set up ..."
		<< std::endl;

	std::cout << GDT_TERMINAL_GREEN;
	std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
	std::cout << GDT_TERMINAL_DEFAULT;
}

void SonelMapVisualizer::setSonelMap(std::vector<OctTree<Sonel>>* sonelMap) {
	this->sonelMap = sonelMap;
	timestep = 0;
}

/*! creates the module that contains all the programs we are going
	to use. in this simple example, we use a single module from a
	single .cu file, using a single embedded ptx string */
void SonelMapVisualizer::createRenderModule() {
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

	const std::string ptxCode = embedded_visualizer_code;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	optixCheck(
		optixModuleCreateFromPTX(
			optixSetup.getOptixContext(), 
			&renderModuleCompileOptions, 
			&renderPipelineCompileOptions,
			ptxCode.c_str(), 
			ptxCode.size(), 
			log, &sizeof_log, 
			&renderModule
		),
		"SonelMapVisualizer",
		"Failed to create render module."
	);
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the raygen program(s) we are going to use */
void SonelMapVisualizer::createRenderRaygenPrograms() {
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
	optixCheck(
		optixProgramGroupCreate(
			optixSetup.getOptixContext(),
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&renderRaygenPgs[0]
		),
		"SonelMapVisualizer",
		"Failed to create raygen program."
	);
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the miss program(s) we are going to use */
void SonelMapVisualizer::createRenderMissPrograms() {
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

	optixCheck(
		optixProgramGroupCreate(
			optixSetup.getOptixContext(), 
			&pgDesc, 
			1, 
			&pgOptions, 
			log,
			&sizeof_log,
			&renderMissPgs[RADIANCE_RAY_TYPE]
		),
		"SonelMapVisualizer",
		"Failed to create miss programs."
	);
	if (sizeof_log > 1)
		PRINT(log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void SonelMapVisualizer::createRenderHitgroupPrograms() {
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

	optixCheck(
		optixProgramGroupCreate(
			optixSetup.getOptixContext(), 
			&pgDesc, 
			1, 
			&pgOptions, 
			log,
			&sizeof_log,
			&renderHitgroupPgs[RADIANCE_RAY_TYPE]
		),
		"SonelMapVisualizer",
		"Failed to create closest and any hit programs."
	);
	if (sizeof_log > 1)
		PRINT(log);
}

/*! assembles the full pipeline of all programs */
void SonelMapVisualizer::createRenderPipeline() {
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

	optixCheck(
		optixPipelineCreate(
			optixSetup.getOptixContext(),
			&renderPipelineCompileOptions,
			&renderPipelineLinkOptions,
			programGroups.data(),
			(int)programGroups.size(),
			log,
			&sizeof_log,
			&renderPipeline
		),
		"SonelMapVisualizer",
		"Failed to create render pipeline"
	);

	if (sizeof_log > 1)
		PRINT(log);

	optixCheck(
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
		),
		"SonelMapVisualizer",
		"Failed to set stack size"
	);

	if (sizeof_log > 1)
		PRINT(log);
}

void SonelMapVisualizer::buildRenderRaygenRecords() {
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

void SonelMapVisualizer::buildRenderMissRecords() {
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

void SonelMapVisualizer::buildRenderHitgroupRecords() {
	const Model* model = cudaScene.getModel();
	int meshSize = model->meshes.size();

	std::vector<HitgroupRecord> hitgroupRecords;
	for (int meshId = 0; meshId < meshSize; meshId++) {
		TriangleMesh* mesh = model->meshes[meshId];

		for (int rayId = 0; rayId < RAY_TYPE_COUNT; rayId++) {
			HitgroupRecord rec;

			optixCheck(
				optixSbtRecordPackHeader(
					renderHitgroupPgs[rayId], 
					&rec
				),
				"SonelMapVisualizer",
				"Failed to create SBT Record Header"
			);
			
			cudaScene.fill(meshId, rec.data);
			hitgroupRecords.push_back(rec);
		}
	}

	renderHitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	renderSbt.hitgroupRecordBase = renderHitgroupRecordsBuffer.d_pointer();
	renderSbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	renderSbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

/*! constructs the shader binding table */
void SonelMapVisualizer::buildRenderSbt() {
	buildRenderRaygenRecords();
	buildRenderMissRecords();
	buildRenderHitgroupRecords();
}

/*! render one frame */
void SonelMapVisualizer::render() {
	// sanity check: make sure we launch only after first resize is
	// already done:
	if (launchParams.frame.size.x == 0) {
		return;
	}

	launchParamsBuffer.upload(&launchParams, 1);

	

	uploadSonelMapSnapshot();
	launchParamsBuffer.upload(&launchParams, 1);

	optixCheck(
		optixLaunch(
			/*! pipeline we're launching launch: */
			renderPipeline,
			optixSetup.getCudaStream(),

			/*! parameters and SBT */
			launchParamsBuffer.d_pointer(),
			launchParamsBuffer.sizeInBytes,
			&renderSbt,

			/*! dimensions of the launch: */
			launchParams.frame.size.x,
			launchParams.frame.size.y,
			1
		),
		"SonelMapVisualizer",
		"Failed to launch OptiX"
	);

	// sync - make sure the frame is rendered before we download and
	// display (obviously, for a high-performance application you
	// want to use streams and double-buffering, but for this simple
	// example, this will have to do)
	cudaSyncCheck("SonelMapVisualizer", "Failed to synchronize.");
}

/*! set camera to render with */
void SonelMapVisualizer::setCamera(const Camera& camera) {
	lastSetCamera = camera;
	launchParams.camera.position = camera.from;
	launchParams.camera.direction = normalize(camera.at - camera.from);

	const float cosFovy = 0.66f;
	const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);

	launchParams.camera.horizontal = cosFovy * aspect * normalize(cross(launchParams.camera.direction, camera.up));
	launchParams.camera.vertical = cosFovy * normalize(cross(launchParams.camera.horizontal, launchParams.camera.direction));
}

/*! resize frame buffer to given resolution */
void SonelMapVisualizer::resize(const vec2i& newSize) {
	// if window minimized
	if (newSize.x == 0 || newSize.y == 0)
		return;

	// resize our cuda frame buffer
	uint64_t bufferSize = (uint64_t)newSize.x * (uint64_t)newSize.y * sizeof(uint32_t);
	colorBuffer.resize(bufferSize);

	// update the launch parameters that we'll pass to the optix
	// launch:
	launchParams.frame.size = newSize;
	launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();

	// and re-set the camera, since aspect may have changed
	setCamera(lastSetCamera);
}

void SonelMapVisualizer::uploadSonelMapSnapshot() {
	printf("Uploading index %d\n", timestep);

	if (sonelMap == nullptr) {
		launchParams.octTree = nullptr;
		return;
	}

	if (launchParams.octTree != nullptr) {
		OctTree<Sonel>::clear(launchParams.octTree);
		cudaFree(launchParams.octTree);
		launchParams.octTree = nullptr;
	}

	OctTree<Sonel>& octTree = (*sonelMap)[timestep];
	OctTree<Sonel>* deviceOctTree = octTree.upload();

	launchParams.octTree = deviceOctTree;

	timestep++;
	if (timestep == sonelMap->size()) {
		timestep = 0;
	}
}

/*! download the rendered color buffer */
void SonelMapVisualizer::downloadPixels(uint32_t h_pixels[]) {
	uint64_t bufferSize = (uint64_t)launchParams.frame.size.x * (uint64_t)launchParams.frame.size.y;

	colorBuffer.download(h_pixels, bufferSize);
}