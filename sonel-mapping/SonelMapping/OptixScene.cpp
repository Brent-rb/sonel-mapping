#include "OptixScene.h"
#include "SonelVisibilityFlags.h"

OptixScene::OptixScene(const OptixDeviceContext& optixContext): 
	optixContext(optixContext), 
	model(nullptr), sonels(nullptr), 
	meshSize(0), sonelSize(0), 
	triangleHandle(0), aabbHandle(0), instanceHandle(0) {

	void* instanceBuffer;
	cudaMalloc(&instanceBuffer, sizeof(OptixInstance) * 2);
	optixInstances.resize(2);
	optixInstanceBuffer = reinterpret_cast<CUdeviceptr>(instanceBuffer);

	float matrix[12] = { 1, 0, 0, 0,
	                     0, 1, 0, 0,
	                     0, 0, 1, 0 };
	optixInstances[0] = {};
	optixInstances[0].flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
	optixInstances[0].instanceId = 0u;
	optixInstances[0].sbtOffset = 0u;
	optixInstances[0].visibilityMask = 255u;

	optixInstances[1] = {};
	optixInstances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
	optixInstances[1].instanceId = 1u;
	optixInstances[0].sbtOffset = 0u;
	optixInstances[1].visibilityMask = 255u;

	memcpy(optixInstances[0].transform, matrix, sizeof(float) * 12);
	memcpy(optixInstances[1].transform, matrix, sizeof(float) * 12);
}

OptixScene::~OptixScene() {

}

void OptixScene::destroy() {
	clear();

	cudaFree(reinterpret_cast<void*>(optixInstanceBuffer));
}

void OptixScene::clear() {
	clearMeshBuffers();
	clearSonelBuffers();
}

void OptixScene::build() {
	cudaSyncCheck("OptixScene", "Sync failed before build.");


	if (meshBuffersInvalid) {
		buildTriangles();
		meshBuffersInvalid = false;
	}
	if (sonelBuffersInvalid) {
		buildSonels();
		sonelBuffersInvalid = false;
	}

	buildInstanceStructure();
}

void OptixScene::buildTriangles() {
	prepareTriangleBuffers();
	buildTriangleInputs();
	buildTextures();
	buildTriangleAccelStructure();
}

void OptixScene::buildSonels() {
	prepareSonelBuffers();
	buildSonelAabbInputs();
	buildSonelAabbAccelStructure();
}

uint32_t OptixScene::getSonelSize() const {
	return sonelSize;
}

void OptixScene::setModel(Model* newModel) {
	clearMeshBuffers();

	this->model = newModel;
	meshSize = static_cast<uint32_t>(newModel->meshes.size());

	meshBuffersInvalid = true;
}

Model* OptixScene::getModel() const {
	return model;
}

void OptixScene::setSonels(std::vector<Sonel>* newSonels, float searchRadius) {
	clearSonelBuffers();
	printf("Setting newSonels %llu\n", newSonels->size());

	this->sonels = newSonels;
	sonelSize = static_cast<uint32_t>(newSonels->size());
	radius = searchRadius;

	sonelBuffersInvalid = true;
}

std::vector<Sonel>* OptixScene::getSonels() const {
	return sonels;
}

CUdeviceptr OptixScene::getSonelDevicePointer(uint32_t sonelIndex) const {
	return sonelBuffer.getCuDevicePointer() + (sonelIndex * sizeof(Sonel));
}

const OptixTraversableHandle& OptixScene::getGeoTraversable() const {
	return triangleHandle;
}

const OptixTraversableHandle& OptixScene::getAabbTraversable() const {
	return aabbHandle;
}

const OptixTraversableHandle& OptixScene::getInstanceTraversable() const {
	return instanceHandle;
}

void OptixScene::fill(const uint32_t meshIndex, TriangleMeshSbtData& triangleData) const {
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

void OptixScene::clearMeshBuffers() {
	printf("Clear mesh buffers\n");
	for (uint32_t meshId = 0; meshId < meshSize; meshId++) {
		// upload the model to the device: the builder
		TriangleMesh& mesh = *model->meshes[meshId];

		vertexBuffer[meshId].tryFree();
		indexBuffer[meshId].tryFree();

		if (!mesh.normal.empty()) {
			normalBuffer[meshId].tryFree();
		}
		if (!mesh.texcoord.empty()) {
			texcoordBuffer[meshId].tryFree();
		}
	}

	triangleAccelBuffer.tryFree();
}

void OptixScene::clearSonelBuffers() {
	aabbAccelBuffer.tryFree();
	sonelBuffer.tryFree();
    sonelAabbBuffer.tryFree();
}

void OptixScene::prepareTriangleBuffers() {
	triangleInputs.resize(meshSize);
	triangleInputFlags.resize(meshSize);

	vertexBuffer.resize(meshSize);
	normalBuffer.resize(meshSize);
	texcoordBuffer.resize(meshSize);
	indexBuffer.resize(meshSize);

	cudaVertices.resize(meshSize);
	cudaIndices.resize(meshSize);
}

void OptixScene::prepareSonelBuffers() {
	aabbInputs.resize(sonelSize);
	aabbInputFlags.resize(sonelSize);

	sonelAabbBuffer.resize(sonelSize);
	sonelBuffer.resize(sonelSize);
	cudaAabbs.resize(sonelSize);
}

void OptixScene::buildTriangleInputs() {	
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

		triangleInputs[meshId] = {};
		triangleInputs[meshId].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// create local variables, because we need a *pointer* to the
		// device pointers
		cudaVertices[meshId] = vertexBuffer[meshId].getCuDevicePointer();
		cudaIndices[meshId] = indexBuffer[meshId].getCuDevicePointer();

		triangleInputs[meshId].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInputs[meshId].triangleArray.vertexStrideInBytes = sizeof(vec3f);
		triangleInputs[meshId].triangleArray.numVertices = (int)mesh.vertex.size();
		triangleInputs[meshId].triangleArray.vertexBuffers = &cudaVertices[meshId];

		triangleInputs[meshId].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInputs[meshId].triangleArray.indexStrideInBytes = sizeof(vec3i);
		triangleInputs[meshId].triangleArray.numIndexTriplets = (int)mesh.index.size();
		triangleInputs[meshId].triangleArray.indexBuffer = cudaIndices[meshId];

		triangleInputFlags[meshId] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

		// in this example we have one SBT entry, and no per-primitive
		// materials:
		triangleInputs[meshId].triangleArray.flags = &triangleInputFlags[meshId];
		triangleInputs[meshId].triangleArray.numSbtRecords = 1;
		triangleInputs[meshId].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInputs[meshId].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInputs[meshId].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}
}


void OptixScene::buildTextures() {
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

void OptixScene::buildSonelAabbInputs() {
	if (sonelSize == 0) {
		return;
	}

	prepareSonelBuffers();
	sonelBuffer.allocAndUpload(*sonels);
    sonelAabbBuffer.alloc(sizeof(OptixAabb) * sonelSize);
    CUdeviceptr aabbDevicePtr = sonelAabbBuffer.getCuDevicePointer();

    std::vector<OptixAabb> tempAabbs;
    tempAabbs.resize(sonelSize);

	for (uint32_t sonelId = 0; sonelId < sonelSize; sonelId++) {
		// upload the model to the device: the builder
		const Sonel& sonel = (*sonels)[sonelId];

		gdt::vec3f min = sonel.position - radius;
		gdt::vec3f max = sonel.position + radius;
		tempAabbs[sonelId] = {
                min.x,
                min.y,
                min.z,
                max.x,
                max.y,
                max.z
        };

        float lengthMin = gdt::length(min - sonel.position);
        float lengthMax = gdt::length(max - sonel.position);
        if (lengthMin > radius * 2 || lengthMax > radius * 2) {
            printf("Length min, max: %f, %f\n", lengthMin, lengthMax);
        }

        cudaAabbs[sonelId] = aabbDevicePtr + (sizeof(OptixAabb) * sonelId);

		aabbInputs[sonelId] = {};
		aabbInputs[sonelId].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

        if (cudaAabbs[sonelId] % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT > 0) {
            printf("Misaligned\n");
        }

		aabbInputs[sonelId].customPrimitiveArray.aabbBuffers = &(cudaAabbs[sonelId]);
		aabbInputs[sonelId].customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
		aabbInputs[sonelId].customPrimitiveArray.numPrimitives = 1;

		aabbInputFlags[sonelId] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;

		aabbInputs[sonelId].customPrimitiveArray.flags = &aabbInputFlags[sonelId];
		aabbInputs[sonelId].customPrimitiveArray.numSbtRecords = 1;
		aabbInputs[sonelId].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
		aabbInputs[sonelId].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
		aabbInputs[sonelId].customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;
	}

    sonelAabbBuffer.upload(tempAabbs.data(), sonelSize);
}

void OptixScene::buildTriangleAccelStructure() {
	this->triangleHandle = buildTraversable(triangleInputs, triangleAccelBuffer);

	optixInstances[0].traversableHandle = this->triangleHandle;
	cudaMemcpy(reinterpret_cast<void*>(optixInstanceBuffer), optixInstances.data(), 2, cudaMemcpyHostToDevice);
}

void OptixScene::buildSonelAabbAccelStructure() {
	if (sonelSize == 0) {
		return;
	}

	this->aabbHandle = buildTraversable(aabbInputs, aabbAccelBuffer);
	optixInstances[1].traversableHandle = this->triangleHandle;

	cudaMemcpy(reinterpret_cast<void*>(optixInstanceBuffer), optixInstances.data(), 2, cudaMemcpyHostToDevice);
}

void OptixScene::buildInstanceStructure() {
	/*
	uint32_t instanceSize = 1;
	printf("Instance size: %d\n", instanceSize);

	OptixBuildInput instanceBuildInput{};
	instanceBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceBuildInput.instanceArray.instances = optixInstanceBuffer;
	instanceBuildInput.instanceArray.numInstances = instanceSize;

	std::vector<OptixBuildInput> buildInputs = { instanceBuildInput };
	
	instanceHandle = buildTraversable(buildInputs, instanceAccelBuffer);
	*/
	float transform[12] = { 1,0,0,3,0,1,0,0,0,0,1,0 };

	OptixInstance instances[2] = { {}, {} };
	memcpy(instances[0].transform, transform, sizeof(float) * 12);
	instances[0].instanceId = 0;
	instances[0].visibilityMask = GEOMETRY_VISIBLE;
	instances[0].sbtOffset = 0;
	instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
	instances[0].traversableHandle = triangleHandle;

	memcpy(instances[1].transform, transform, sizeof(float) * 12);
	instances[1].instanceId = 1;
	instances[1].visibilityMask = SONELS_VISIBLE;
	instances[1].sbtOffset = meshSize;
	instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
	instances[1].traversableHandle = aabbHandle;


	void* d_instance;

	cudaMalloc(&d_instance, sizeof(OptixInstance) * 2);


	cudaMemcpy(d_instance, instances, sizeof(OptixInstance) * 2, cudaMemcpyHostToDevice);

	std::vector<OptixBuildInput> buildInputs;
	buildInputs.resize(1);
	OptixBuildInput& buildInput = buildInputs[0];
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	buildInput.instanceArray.instances = reinterpret_cast<CUdeviceptr>(d_instance);
	buildInput.instanceArray.numInstances = 2;

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
