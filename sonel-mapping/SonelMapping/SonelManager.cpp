#include "SonelManager.h"
#include <optix_function_table_definition.h>
#include "../Cuda/CudaRandom.h"
#include <chrono>

using namespace std::chrono;

SonelManager::SonelManager(
		Model* model,
		SonelMapperConfig sonelMapperConfig
): optixSetup(), 
	optixScene(optixSetup.getOptixContext()),
	sonelMapper(
		optixSetup,
		optixScene
	),
	sonelVisualizer(
		optixSetup,
		optixScene,
        sonelMapperConfig.timestep,
        0.15f
	),
   sonelMapReceiver(
	   optixSetup,
	   optixScene
   ), absorptionData(50) {
	auto sec_since_epoch = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
	std::stringstream outputName;
	outputName << "C:\\Users\\brent\\Desktop\\timings\\timing_" << sec_since_epoch << ".csv";
	timingFile.open(outputName.str());
	timingFile.imbue(std::locale("be"));

	auto modelStart = high_resolution_clock::now();
	optixScene.setModel(model);
	optixScene.build();
	auto modelEnd = high_resolution_clock::now();
	auto modelDelta = modelEnd - modelStart;
	auto modelMs = duration_cast<microseconds>(modelDelta);
	timingFile << modelMs.count() / 1000.0f << "\t";
	printf("[Time][SonelManager][Model Build] %f\n", modelMs.count() / 1000.0f);

	auto initStart = high_resolution_clock::now();
	sonelMapper.initialize(sonelMapperConfig);
	SimulationData& simulationData = sonelMapper.getSimulationData();
	absorptionData.setAbsorptions(simulationData.frequencies, simulationData.frequencySize);
	sonelMapReceiver.initialize({
		10000,
		sonelMapperConfig.soundSpeed,
		sonelMapperConfig.echogramDuration,
		sonelMapperConfig.timestep,
		sonelMapper.getSimulationData().frequencySize,
		static_cast<uint32_t>(round(sonelMapperConfig.echogramDuration / sonelMapperConfig.timestep)),
		1,
		&simulationData,
		&absorptionData
	});

	auto initEnd = high_resolution_clock::now();
	auto initDelta = initEnd - initStart;
	auto initMs = duration_cast<microseconds>(initDelta);
	timingFile << initMs.count() / 1000.0f << "\t";
	printf("[Time][SonelManager][Init] %f\n", initMs.count() / 1000.0f);

	sonelVisualizer.initialize();
}

SonelManager::~SonelManager() {
}

void SonelManager::calculate() {
	auto managerStart = high_resolution_clock::now();

	auto mapperStart = high_resolution_clock::now();
	sonelMapper.execute(timingFile);
	auto mapperEnd = high_resolution_clock::now();
	auto mapperDelta = mapperEnd - mapperStart;
	auto mapperMs = duration_cast<microseconds>(mapperDelta);
	timingFile << mapperMs.count() / 1000.0f << "\t";
	printf("[Time][SonelManager][SonelMapper] %f\n", mapperMs.count() / 1000.0f);


	auto prepStart = high_resolution_clock::now();
	std::vector<Sonel>* sonels = sonelMapper.getSonelArray();
	sonelMapReceiver.setSonels(sonels);
	simpleSoundSources = SimpleSoundSource::from(sonelMapper.getSimulationData());
	optixScene.setSonels(sonels, 0.15f);
	optixScene.setSoundSources(&simpleSoundSources);
    optixScene.build();
	auto prepEnd = high_resolution_clock::now();
	auto prepDelta = prepEnd - prepStart;
	auto prepMs = duration_cast<microseconds>(prepDelta);
	timingFile << prepMs.count() / 1000.0f << "\t";
	printf("[Time][SonelManager][Gather Prep] %f\n", prepMs.count() / 1000.0f);

	auto receiverStart = high_resolution_clock::now();
	sonelMapReceiver.execute(timingFile);
	auto receiverEnd = high_resolution_clock::now();
	auto receiverDelta = receiverEnd - receiverStart;
	auto receiverMs = duration_cast<microseconds>(receiverDelta);
	timingFile << receiverMs.count() / 1000.0f << "\t";
	printf("[Time][SonelManager][Receiver] %f\n", receiverMs.count() / 1000.0f);

	auto managerEnd = high_resolution_clock::now();
	auto managerDelta = managerEnd - managerStart;
	auto managerMs = duration_cast<microseconds>(managerDelta);
	timingFile << managerMs.count() / 1000.0f << "\t";
	printf("[Time][SonelManager][Total] %f\n", managerMs.count() / 1000.0f);
	timingFile.close();

	exit(EXIT_SUCCESS);
    // sonelVisualizer.setFrequencySize(sonelMapper.getSimulationData().frequencySize);
	// sonelVisualizer.setSonelArray(sonels);
}

void SonelManager::render() {
	// sonelVisualizer.execute();
}

void SonelManager::resize(const vec2i& newSize) {
	// sonelVisualizer.resize(newSize);
}

void SonelManager::downloadPixels(uint32_t h_pixels[]) {
	// sonelVisualizer.downloadPixels(h_pixels);
}

void SonelManager::setCamera(const Camera& camera) {
	sonelMapReceiver.setCamera(camera);
	sonelVisualizer.setCamera(camera);
}

void SonelManager::nextFrame() {
    sonelVisualizer.nextFrame();
}

void SonelManager::previousFrame() {
    sonelVisualizer.previousFrame();
}
