#include "OptixScene.h"
#include "SonelVisibilityFlags.h"

OptixScene::OptixScene(const OptixDeviceContext& optixContext):
		optixContext(optixContext),
		model(nullptr), sonels(nullptr), soundSources(nullptr),
		meshSize(0), sonelSize(0), soundSourceSize(0),
		geometryHandle(0), sonelHandle(0), soundSourceHandle(0), instanceHandle(0) {

	void* instanceBuffer;
	cudaMalloc(&instanceBuffer, sizeof(OptixInstance) * instanceSize);
	optixInstances.resize(instanceSize);
	optixInstanceBuffer = reinterpret_cast<CUdeviceptr>(instanceBuffer);

	float matrix[12] = { 1, 0, 0, 0,
	                     0, 1, 0, 0,
	                     0, 0, 1, 0 };

	for (uint32_t i = 0; i < instanceSize; i++) {
		optixInstances[i] = {};
		optixInstances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
		optixInstances[i].instanceId = i;
		optixInstances[i].sbtOffset = 0u;
		optixInstances[i].visibilityMask = 255u;
		memcpy(optixInstances[i].transform, matrix, sizeof(float) * 12);
	}

	optixInstances[0].visibilityMask = GEOMETRY_VISIBLE;
	optixInstances[1].visibilityMask = SONELS_VISIBLE;
	optixInstances[2].visibilityMask = SOUND_SOURCES_VISIBLE;
}

OptixScene::~OptixScene() {

}

void OptixScene::destroy() {
	clear();

	cudaFree(reinterpret_cast<void*>(optixInstanceBuffer));
}

void OptixScene::clear() {
	clearGeometryBuffers();
	clearSonelBuffers();
	clearSoundSourceBuffers();
}

void OptixScene::build() {
	cudaSyncCheck("OptixScene", "Sync failed before build.");


	if (geometryBuffersInvalid) {
		buildGeometry();
		geometryBuffersInvalid = false;
	}
	if (sonelBuffersInvalid) {
		buildSonels();
		sonelBuffersInvalid = false;
	}
	if (soundSourceBuffersInvalid) {
		buildSoundSources();
		soundSourceBuffersInvalid = false;
	}

	buildInstanceAccelStructure();
}

void OptixScene::buildGeometry() {
	clearGeometryBuffers();
	prepareGeometryBuffers();
	buildGeometryInputs();
	buildGeometryTextures();
	buildGeometryAccelStructure();
}

void OptixScene::buildSonels() {
	clearSonelBuffers();
	prepareSonelBuffers();
	buildSonelInputs();
	buildSonelAccelStructure();
}

void OptixScene::buildSoundSources() {
	clearSoundSourceBuffers();
	prepareSoundSourceBuffers();
	buildSoundSourceInputs();
	buildSoundSourceAccelStructure();
}

uint32_t OptixScene::getSonelSize() const {
	return sonelSize;
}

uint32_t OptixScene::getSoundSourceSize() const {
	return soundSourceSize;
}

void OptixScene::setModel(Model* newModel) {
	printf("[OptixScene] Setting model, meshes: %llu\n", newModel->meshes.size());

	model = newModel;
	meshSize = static_cast<uint32_t>(newModel->meshes.size());

	geometryBuffersInvalid = true;
}

Model* OptixScene::getModel() const {
	return model;
}

void OptixScene::setSonels(std::vector<Sonel>* newSonels, float searchRadius) {
	printf("[OptixScene] Setting sonels, size: %llu\n", newSonels->size());

	sonels = newSonels;
	sonelSize = static_cast<uint32_t>(newSonels->size());
	sonelRadius = searchRadius;

	sonelBuffersInvalid = true;
}

std::vector<Sonel>* OptixScene::getSonels() const {
	return sonels;
}

void OptixScene::setSoundSources(std::vector<SimpleSoundSource> *newSoundSources, float searchRadius) {
	printf("[OptixScene] Setting sound sources, size: %llu\n", newSoundSources->size());

	soundSources = newSoundSources;
	soundSourceSize = static_cast<uint32_t>(newSoundSources->size());
	soundSourceRadius = searchRadius;

	soundSourceBuffersInvalid = true;
}

std::vector<SimpleSoundSource> *OptixScene::getSoundSources() const {
	return soundSources;
}

CUdeviceptr OptixScene::getSonelDevicePointer(uint32_t sonelIndex) const {
	return getDevicePtr<Sonel>(sonelBuffer, sonelIndex);
}

CUdeviceptr OptixScene::getSoundSourceDevicePointer(uint32_t soundSourceIndex) const {
	return getDevicePtr<SimpleSoundSource>(soundSourceBuffer, soundSourceIndex);
}

const OptixTraversableHandle& OptixScene::getGeometryHandle() const {
	return geometryHandle;
}

const OptixTraversableHandle& OptixScene::getSonelHandle() const {
	return sonelHandle;
}

const OptixTraversableHandle& OptixScene::getInstanceHandle() const {
	return instanceHandle;
}

void OptixScene::fill(const uint32_t meshIndex, SmSbtData& triangleData) const {
	TriangleMesh* mesh = model->meshes[meshIndex];

	// Load color data
	triangleData.color = mesh->diffuse;
	if (mesh->diffuseTextureId >= 0) {
		triangleData.hasTexture = true;
		triangleData.texture = textureObjects[mesh->diffuseTextureId];
	}
	else {
		triangleData.hasTexture = false;
	}

	// Load vector data
	triangleData.index = (vec3i*)indexBuffer[meshIndex].getCuDevicePointer();
	triangleData.vertex = (vec3f*)vertexBuffer[meshIndex].getCuDevicePointer();
	triangleData.normal = (vec3f*)normalBuffer[meshIndex].getCuDevicePointer();
	triangleData.texcoord = (vec2f*)texcoordBuffer[meshIndex].getCuDevicePointer();
}

void OptixScene::clearGeometryBuffers() {
	printf("[OptixScene] Clearing geometry buffers.\n");
	for (uint32_t meshId = 0; meshId < vertexBuffer.size(); meshId++) {
		vertexBuffer[meshId].tryFree();
		indexBuffer[meshId].tryFree();
		normalBuffer[meshId].tryFree();
		texcoordBuffer[meshId].tryFree();
	}

	meshAccelBuffer.tryFree();
}

void OptixScene::clearSonelBuffers() {
	printf("[OptixScene] Clearing sonel buffers.\n");
	sonelAccelBuffer.tryFree();
	sonelBuffer.tryFree();
    sonelAabbBuffer.tryFree();
}

void OptixScene::clearSoundSourceBuffers() {
	printf("[OptixScene] Clearing sound source buffers.\n");
	soundSourceAccelBuffer.tryFree();
	soundSourceBuffer.tryFree();
	soundSourceAabbBuffer.tryFree();
}

void OptixScene::prepareGeometryBuffers() {
	geometryInputs.resize(meshSize);
	geometryInputFlags.resize(meshSize);

	vertexBuffer.resize(meshSize);
	normalBuffer.resize(meshSize);
	texcoordBuffer.resize(meshSize);
	indexBuffer.resize(meshSize);

	cudaGeometryVertices.resize(meshSize);
	cudaGeometryIndices.resize(meshSize);
}

void OptixScene::prepareSonelBuffers() {
	sonelInputs.resize(sonelSize);
	sonelInputFlags.resize(sonelSize);

	sonelAabbBuffer.resize(sonelSize);
	sonelBuffer.resize(sonelSize);
	cudaSonelInputs.resize(sonelSize);
}

void OptixScene::prepareSoundSourceBuffers() {
	soundSourceInputs.resize(soundSourceSize);
	soundSourceInputFlags.resize(soundSourceSize);

	soundSourceAabbBuffer.resize(soundSourceSize);
	soundSourceBuffer.resize(soundSourceSize);
	cudaSoundSourceInputs.resize(soundSourceSize);
}

void OptixScene::buildGeometryInputs() {
	for (uint32_t meshId = 0; meshId < meshSize; meshId++) {
		// upload the model to the device: the builder
		TriangleMesh& mesh = *model->meshes[meshId];

		vertexBuffer[meshId].allocAndUpload(mesh.vertex);
		indexBuffer[meshId].allocAndUpload(mesh.index);

		if (!mesh.normal.empty()) {
			normalBuffer[meshId].allocAndUpload(mesh.normal);
		}
		if (!mesh.texcoord.empty()) {
			texcoordBuffer[meshId].allocAndUpload(mesh.texcoord);
		}

		geometryInputs[meshId] = {};
		geometryInputs[meshId].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// create local variables, because we need a *pointer* to the
		// device pointers
		cudaGeometryVertices[meshId] = vertexBuffer[meshId].getCuDevicePointer();
		cudaGeometryIndices[meshId] = indexBuffer[meshId].getCuDevicePointer();

		geometryInputs[meshId].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		geometryInputs[meshId].triangleArray.vertexStrideInBytes = sizeof(vec3f);
		geometryInputs[meshId].triangleArray.numVertices = (int)mesh.vertex.size();
		geometryInputs[meshId].triangleArray.vertexBuffers = &cudaGeometryVertices[meshId];

		geometryInputs[meshId].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		geometryInputs[meshId].triangleArray.indexStrideInBytes = sizeof(vec3i);
		geometryInputs[meshId].triangleArray.numIndexTriplets = (int)mesh.index.size();
		geometryInputs[meshId].triangleArray.indexBuffer = cudaGeometryIndices[meshId];

		geometryInputFlags[meshId] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

		// in this example we have one SBT entry, and no per-primitive
		// materials:
		geometryInputs[meshId].triangleArray.flags = &geometryInputFlags[meshId];
		geometryInputs[meshId].triangleArray.numSbtRecords = 1;
		geometryInputs[meshId].triangleArray.sbtIndexOffsetBuffer = 0;
		geometryInputs[meshId].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		geometryInputs[meshId].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}
}


void OptixScene::buildGeometryTextures() {
	auto numTextures = static_cast<uint32_t>(model->textures.size());

	textureArrays.resize(numTextures);
	textureObjects.resize(numTextures);

	for (uint32_t textureId = 0; textureId < numTextures; textureId++) {
		auto texture = model->textures[textureId];

		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc = {};
		int32_t width = texture->resolution.x;
		int32_t height = texture->resolution.y;
		int32_t numComponents = 4;
		int32_t pitch = width * numComponents * sizeof(uint8_t);
		channel_desc = cudaCreateChannelDesc<uchar4>();

		cudaArray_t& pixelArray = textureArrays[textureId];
		cudaCheck(
			cudaMallocArray(&pixelArray, &channel_desc, width, height),
			"CudaScene",
			"Failed to allocate pixel array."
		);

		cudaCheck(
			cudaMemcpy2DToArray(pixelArray, 0, 0, texture->pixel, pitch, pitch, height, cudaMemcpyHostToDevice),
			"CudaScene",
			"Failed to copy texture data."
		);

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
		cudaCheck(
			cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr),
			"CudaScene",
			"Failed to create texture object"
		);
		textureObjects[textureId] = cuda_tex;
	}
}

void OptixScene::buildSonelInputs() {
	buildAabb(sonels, sonelBuffer, sonelAabbBuffer,
			  cudaSonelInputs, sonelInputs, sonelInputFlags,
			  sonelRadius);
}

void OptixScene::buildSoundSourceInputs() {
	buildAabb(soundSources, soundSourceBuffer, soundSourceAabbBuffer,
			  cudaSoundSourceInputs, soundSourceInputs, soundSourceInputFlags,
			  soundSourceRadius);
}

void OptixScene::buildGeometryAccelStructure() {
	geometryHandle = buildTraversable(geometryInputs, meshAccelBuffer);
	optixInstances[0].traversableHandle = geometryHandle;
}

void OptixScene::buildSonelAccelStructure() {
	if (sonelSize == 0) {
		sonelHandle = 0;
		return;
	}

	sonelHandle = buildTraversable(sonelInputs, sonelAccelBuffer);
	optixInstances[1].sbtOffset = meshSize;
	optixInstances[1].traversableHandle = sonelHandle;
}

void OptixScene::buildSoundSourceAccelStructure() {
	if (soundSourceSize == 0) {
		soundSourceHandle = 0;
		return;
	}

	soundSourceHandle = buildTraversable(soundSourceInputs, soundSourceAabbBuffer);
	optixInstances[2].sbtOffset = meshSize + sonelSize;
	optixInstances[2].traversableHandle = soundSourceHandle;
}

void OptixScene::buildInstanceAccelStructure() {
	OptixInstance instances[3] = { optixInstances[0], optixInstances[1], optixInstances[2] };

	void* deviceInstances = reinterpret_cast<void*>(optixInstanceBuffer);
	cudaMemcpy(deviceInstances, instances, sizeof(OptixInstance) * instanceSize, cudaMemcpyHostToDevice);

	std::vector<OptixBuildInput> buildInputs;
	buildInputs.resize(1);
	OptixBuildInput& buildInput = buildInputs[0];
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	buildInput.instanceArray.instances = reinterpret_cast<CUdeviceptr>(deviceInstances);
	buildInput.instanceArray.numInstances = instanceSize;

	instanceHandle = buildTraversable(buildInputs, instanceAccelBuffer);
}

OptixTraversableHandle OptixScene::buildTraversable(const std::vector<OptixBuildInput>& buildInputs, CudaBuffer& cudaBuffer) {
	return buildTraversable(optixContext, buildInputs, cudaBuffer);
}

OptixTraversableHandle OptixScene::buildTraversable(const OptixDeviceContext& optixContext, const std::vector<OptixBuildInput>& buildInputs, CudaBuffer& accelBuffer) {
	OptixTraversableHandle traversableHandle = { 0 };
	OptixAccelBufferSizes bufferSizes;

	// BLAS Setup
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	optixCheck(
		optixAccelComputeMemoryUsage(
			optixContext,
			&accelOptions,
			buildInputs.data(),
			static_cast<uint32_t>(buildInputs.size()),
			&bufferSizes
		),
		"CudaScene",
		"Failed to compute acceleration structure memory usage."
	);

	CudaBuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.getCuDevicePointer();

	CudaBuffer tempBuffer;
	tempBuffer.alloc(bufferSizes.tempSizeInBytes);

	CudaBuffer outputBuffer;
	outputBuffer.alloc(bufferSizes.outputSizeInBytes);

	optixCheck(
		optixAccelBuild(
			optixContext,
            nullptr,
			&accelOptions,
			buildInputs.data(), (int) buildInputs.size(),
			tempBuffer.getCuDevicePointer(), tempBuffer.sizeInBytes,
			outputBuffer.getCuDevicePointer(), outputBuffer.sizeInBytes,

			&traversableHandle,
			&emitDesc, 1
		),
		"CudaScene",
		"Failed to build acceleration structure."
	);

	// Compact
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	accelBuffer.alloc(compactedSize);
	optixCheck(
		optixAccelCompact(
			optixContext,
			nullptr,
			traversableHandle,
			accelBuffer.getCuDevicePointer(), accelBuffer.sizeInBytes,
			&traversableHandle
		),
		"CudaScene",
		"Failed to compact acceleration structure."
	);
	cudaSyncCheck("CudaScene", "Failed to synchronize after acceleration structure build.");

	// Clean up
	outputBuffer.free();
	tempBuffer.free();
	compactedSizeBuffer.free();

	return traversableHandle;
}