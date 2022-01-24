#include "SonelMapVisualizer.h"

#include "LaunchParams.h"
#include "OctTree.h"
#include "TriangleMeshSbtData.h"
#include "CudaHelper.h"
#include <chrono>

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
	OptixScene& cudaScene
): optixSetup(optixSetup), cudaScene(cudaScene), sonelMap(nullptr), sonelArray(nullptr) {
	
}

void SonelMapVisualizer::init() {
	std::cout << "[SonelMapVisualizer] Creating render module." << std::endl;
	createRenderModule();

	std::cout << "[SonelMapVisualizer] Creating raygen programs." << std::endl;
	createRenderRaygenPrograms();

	std::cout << "[SonelMapVisualizer] Creating miss programs." << std::endl;
	createRenderMissPrograms();

	std::cout << "[SonelMapVisualizer] Creating hitgroup programs." << std::endl;
	createRenderHitgroupPrograms();

	std::cout << "[SonelMapVisualizer] Creating render pipeline." << std::endl;
	createRenderPipeline();


	std::cout << "[SonelMapVisualizer] Building SBT." << std::endl;
	buildRenderSbt();

	launchParamsBuffer.alloc(sizeof(launchParams));
	std::cout << "[SonelMapVisualizer] context, module, pipeline, etc, all set up ..."
		<< std::endl;

	std::cout << GDT_TERMINAL_GREEN;
	std::cout << "[SonelMapVisualizer] Optix 7 Sample fully set up" << std::endl;
	std::cout << GDT_TERMINAL_DEFAULT;
}

void SonelMapVisualizer::setSonelMap(std::vector<OctTree<Sonel>>* sonelMap) {
	this->sonelMap = sonelMap;
	timestep = 0;
}

void SonelMapVisualizer::setSonelArray(std::vector<std::vector<Sonel>>* sonelArray) {
	this->sonelArray = sonelArray;
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
	renderPipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	renderPipelineCompileOptions.usesMotionBlur = false;
	renderPipelineCompileOptions.numPayloadValues = 2;
	renderPipelineCompileOptions.numAttributeValues = 2;
	renderPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	renderPipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
	renderPipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

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
	pgDesc.hitgroup.moduleIS = renderModule;

	// -------------------------------------------------------
	// radiance rays
	// -------------------------------------------------------
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
	pgDesc.hitgroup.entryFunctionNameIS = "__intersection__radiance";

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
			3
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
		optixCheck(
			optixSbtRecordPackHeader(renderRaygenPgs[i], &rec),
			"SonelMapVisualizer",
			"Failed to record pack header (render raygen pgs)."
		);
		rec.data = nullptr; /* for now ... */
		raygenRecords.push_back(rec);
	}

	renderRaygenRecordsBuffer.allocAndUpload(raygenRecords);
	renderSbt.raygenRecord = renderRaygenRecordsBuffer.getCuDevicePointer();
}

void SonelMapVisualizer::buildRenderMissRecords() {
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < renderMissPgs.size(); i++) {
		MissRecord rec;
		optixCheck(
			optixSbtRecordPackHeader(renderMissPgs[i], &rec),
			"SonelMapVisualizer",
			"Failed to record pack header (render miss pgs)."
		);
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}

	renderMissRecordsBuffer.allocAndUpload(missRecords);
	renderSbt.missRecordBase = renderMissRecordsBuffer.getCuDevicePointer();

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
			rec.data.sonel = nullptr;
			hitgroupRecords.push_back(rec);
		}
	}

	for (int sonelId = 0; sonelId < cudaScene.getSonelSize(); sonelId++) {
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

			rec.data.sonel = (Sonel*) cudaScene.getSonelDevicePointer(sonelId);
			hitgroupRecords.push_back(rec);
		}
	}

	renderHitgroupRecordsBuffer.allocAndUpload(hitgroupRecords);
	renderSbt.hitgroupRecordBase = renderHitgroupRecordsBuffer.getCuDevicePointer();
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
	
	uploadSonelMapSnapshot();
	launchParams.traversable = cudaScene.getInstanceTraversable();
	launchParamsBuffer.upload(&launchParams, 1);

	auto start = std::chrono::high_resolution_clock::now();
	optixCheck(
		optixLaunch(
			/*! pipeline we're launching launch: */
			renderPipeline,
			optixSetup.getCudaStream(),

			/*! parameters and SBT */
			launchParamsBuffer.getCuDevicePointer(),
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

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	printf("[SonelMapVisualizer] Rendering frame took %lf ms\n", duration.count() / 1000.0);
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
	launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.getCuDevicePointer();

	// and re-set the camera, since aspect may have changed
	setCamera(lastSetCamera);
}

void SonelMapVisualizer::uploadSonelMapSnapshot() {
	printf("[SonelMapVisualizer] Uploading Sonel index %d\n", timestep);

	auto start = std::chrono::high_resolution_clock::now();
	if (sonelArray != nullptr) {
		std::vector<Sonel>& sonels = (*sonelArray)[timestep];
		if (sonels.size() > 0) {
			cudaScene.setSonels(&sonels, 5.0f);
			cudaScene.build();
			buildRenderHitgroupRecords();
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	printf("[SonelMapVisualizer] Uploading took %lf ms\n", duration.count() / 1000.0);

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