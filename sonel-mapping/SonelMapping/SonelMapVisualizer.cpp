#include "SonelMapVisualizer.h"
#include <chrono>

extern "C" char embedded_visualizer_code[];

/*! constructor - performs all setup, including initializing
	optix, creates module, pipeline, programs, SBT, etc. */
SonelMapVisualizer::SonelMapVisualizer(
	const OptixSetup& optixSetup,
	OptixScene& cudaScene
): SmOptixProgram<LaunchParams, EmptyRecord, EmptyRecord, TriangleMeshSbtData>(embedded_visualizer_code, optixSetup, cudaScene, 1, 1, 1), cudaScene(cudaScene), sonelArray(nullptr) {
	maxTraversableGraphDepth = 3;
	hitIsEnabled = true;
	hitAhEnabled = true;
}

void SonelMapVisualizer::setSonelArray(std::vector<std::vector<Sonel>>* newSonelArray) {
	this->sonelArray = newSonelArray;
	timestep = 0;
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
			createHitRecords();
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	printf("[SonelMapVisualizer] Uploading took %lf ms\n", duration.count() / 1000.0);

	timestep++;
	if (timestep == sonelArray->size()) {
		timestep = 0;
	}
}

/*! download the rendered color buffer */
void SonelMapVisualizer::downloadPixels(uint32_t h_pixels[]) {
	uint64_t bufferSize = (uint64_t)launchParams.frame.size.x * (uint64_t)launchParams.frame.size.y;

	colorBuffer.download(h_pixels, bufferSize);
}

void SonelMapVisualizer::setFrequencySize(uint32_t size) {
    launchParams.frequencySize = size;
}

void SonelMapVisualizer::initialize() {
	init();
}

void SonelMapVisualizer::execute() {
	// sanity check: make sure we launch only after first resize is
	// already done:
	if (launchParams.frame.size.x == 0) {
		return;
	}

	uploadSonelMapSnapshot();
	launchParams.traversable = cudaScene.getInstanceTraversable();

	auto start = std::chrono::high_resolution_clock::now();
	launchOptix(launchParams.frame.size.x,launchParams.frame.size.y,1);

	// sync - make sure the frame is rendered before we download and
	// display (obviously, for a high-performance application you
	// want to use streams and double-buffering, but for this simple
	// example, this will have to do)
	cudaSyncCheck("SonelMapVisualizer", "Failed to synchronize.");

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	printf("[SonelMapVisualizer] Rendering frame took %lf ms\n", duration.count() / 1000.0);
}

const char *SonelMapVisualizer::getLaunchParamsName() {
	return "launchParams";
}

void SonelMapVisualizer::configureModuleCompileOptions(OptixModuleCompileOptions &compileOptions) {
	SmOptixProgram::configureModuleCompileOptions(compileOptions);
}

void SonelMapVisualizer::configurePipelineCompileOptions(OptixPipelineCompileOptions &pipelineOptions) {
	SmOptixProgram::configurePipelineCompileOptions(pipelineOptions);

	pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;\
	pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
}

void SonelMapVisualizer::configurePipelineLinkOptions(OptixPipelineLinkOptions &pipelineLinkOptions) {
	SmOptixProgram::configurePipelineLinkOptions(pipelineLinkOptions);
}

void SonelMapVisualizer::configureRaygenProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.raygen.entryFunctionName = "__raygen__renderFrame";
}

void SonelMapVisualizer::configureMissProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.miss.entryFunctionName = "__miss__radiance";
}

void SonelMapVisualizer::configureHitProgram(
		uint32_t programIndex, OptixProgramGroupOptions &options, OptixProgramGroupDesc &desc
) {
	desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
	desc.hitgroup.entryFunctionNameIS = "__intersection__radiance";
}

void SonelMapVisualizer::addHitRecords(std::vector<SmRecord<TriangleMeshSbtData>> &hitRecords) {
	const Model* model = cudaScene.getModel();
	auto meshSize = static_cast<uint32_t>(model->meshes.size());

	for (uint32_t meshId = 0; meshId < meshSize; meshId++) {
		TriangleMesh* mesh = model->meshes[meshId];

		for (uint32_t programIndex = 0; programIndex < hitgroupProgramSize; programIndex++) {
			SmRecord<TriangleMeshSbtData> rec;

			optixCheck(
				optixSbtRecordPackHeader(
					hitgroupPgs[programIndex],
					&rec
				),
				"SonelMapVisualizer",
				"Failed to create SBT Record Header"
			);

			cudaScene.fill(meshId, rec.data);
			rec.data.sonel = nullptr;
			hitRecords.push_back(rec);
		}
	}

	for (uint32_t sonelId = 0; sonelId < static_cast<uint32_t>(cudaScene.getSonelSize()); sonelId++) {
		for (uint32_t programIndex = 0; programIndex < hitgroupProgramSize; programIndex++) {
			SmRecord<TriangleMeshSbtData> rec;

			optixCheck(
				optixSbtRecordPackHeader(
					hitgroupPgs[programIndex],
					&rec
				),
				"SonelMapVisualizer",
				"Failed to create SBT Record Header"
			);

			rec.data.sonel = (Sonel*) cudaScene.getSonelDevicePointer(sonelId);
			hitRecords.push_back(rec);
		}
	}
}
