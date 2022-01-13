#include "OptixScene.h"
#include "CudaHelper.h"

OptixScene::OptixScene(const OptixDeviceContext& optixContext, const Model* model): model(model), meshSize(model->meshes.size()), traversableHandle(0) {
	build(optixContext);
}

OptixScene::~OptixScene() {
	for (int meshId = 0; meshId < meshSize; meshId++) {
		// upload the model to the device: the builder
		TriangleMesh& mesh = *model->meshes[meshId];

		vertexBuffer[meshId].free();
		indexBuffer[meshId].free();

		if (!mesh.normal.empty()) {
			normalBuffer[meshId].free();
		}
		if (!mesh.texcoord.empty()) {
			texcoordBuffer[meshId].free();
		}
	}

	accelBuffer.free();
}

const OptixTraversableHandle& OptixScene::getTraversableHandle() const {
	return traversableHandle;
}

const Model* OptixScene::getModel() const {
	return model;
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
	triangleData.index = (vec3i*)indexBuffer[meshIndex].d_pointer();
	triangleData.vertex = (vec3f*)vertexBuffer[meshIndex].d_pointer();
	triangleData.normal = (vec3f*)normalBuffer[meshIndex].d_pointer();
	triangleData.texcoord = (vec2f*)texcoordBuffer[meshIndex].d_pointer();
}

void OptixScene::build(const OptixDeviceContext& optixContext) {
	meshSize = model->meshes.size();

	OptixTraversableHandle accelHandle{ 0 };

	// Triangle inputs
	std::vector<OptixBuildInput> triangleInputs;
	std::vector<uint32_t> triangleInputFlags;
	std::vector<CUdeviceptr> cudaVertices;
	std::vector<CUdeviceptr> cudaIndices;

	prepareBuffers(triangleInputs, triangleInputFlags, cudaVertices, cudaIndices);
	buildTriangleInput(triangleInputs, triangleInputFlags, cudaVertices, cudaIndices);
	buildAccelStructure(optixContext, triangleInputs, accelHandle);
	buildTextures();

	this->traversableHandle = accelHandle;
}

void OptixScene::prepareBuffers(
	std::vector<OptixBuildInput>& triangleInputs,
	std::vector<uint32_t>& triangleInputFlags,
	std::vector<CUdeviceptr>& cudaVertices,
	std::vector<CUdeviceptr>& cudaIndices
) {
	vertexBuffer.resize(meshSize);
	normalBuffer.resize(meshSize);
	texcoordBuffer.resize(meshSize);
	indexBuffer.resize(meshSize);

	triangleInputs.resize(meshSize);
	triangleInputFlags.resize(meshSize);
	cudaVertices.resize(meshSize);
	cudaIndices.resize(meshSize);
}

void OptixScene::buildTriangleInput(
	std::vector<OptixBuildInput>& triangleInputs, 
	std::vector<uint32_t>& triangleInputFlags, 
	std::vector<CUdeviceptr>& cudaVertices, 
	std::vector<CUdeviceptr>& cudaIndices
) {	
	for (int meshId = 0; meshId < meshSize; meshId++) {
		// upload the model to the device: the builder
		TriangleMesh& mesh = *model->meshes[meshId];

		vertexBuffer[meshId].alloc_and_upload(mesh.vertex);
		indexBuffer[meshId].alloc_and_upload(mesh.index);

		if (!mesh.normal.empty()) {
			normalBuffer[meshId].alloc_and_upload(mesh.normal);
		}
		if (!mesh.texcoord.empty()) {
			texcoordBuffer[meshId].alloc_and_upload(mesh.texcoord);
		}

		triangleInputs[meshId] = {};
		triangleInputs[meshId].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// create local variables, because we need a *pointer* to the
		// device pointers
		cudaVertices[meshId] = vertexBuffer[meshId].d_pointer();
		cudaIndices[meshId] = indexBuffer[meshId].d_pointer();

		triangleInputs[meshId].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInputs[meshId].triangleArray.vertexStrideInBytes = sizeof(vec3f);
		triangleInputs[meshId].triangleArray.numVertices = (int)mesh.vertex.size();
		triangleInputs[meshId].triangleArray.vertexBuffers = &cudaVertices[meshId];

		triangleInputs[meshId].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInputs[meshId].triangleArray.indexStrideInBytes = sizeof(vec3i);
		triangleInputs[meshId].triangleArray.numIndexTriplets = (int)mesh.index.size();
		triangleInputs[meshId].triangleArray.indexBuffer = cudaIndices[meshId];

		triangleInputFlags[meshId] = 0;

		// in this example we have one SBT entry, and no per-primitive
		// materials:
		triangleInputs[meshId].triangleArray.flags = &triangleInputFlags[meshId];
		triangleInputs[meshId].triangleArray.numSbtRecords = 1;
		triangleInputs[meshId].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInputs[meshId].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInputs[meshId].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}
}

void OptixScene::buildAccelStructure(
	const OptixDeviceContext& optixContext,
	const std::vector<OptixBuildInput>& triangleInputs,
	OptixTraversableHandle& accelHandle
) {
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
			triangleInputs.data(),
			(int)model->meshes.size(), // num_build_inputs
			&bufferSizes
		),
		"CudaScene",
		"Failed to compute acceleration structure memory usage."
	);

	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	CUDABuffer tempBuffer;
	tempBuffer.alloc(bufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(bufferSizes.outputSizeInBytes);

	optixCheck(
		optixAccelBuild(
			optixContext,
			0,
			&accelOptions,
			triangleInputs.data(), (int)meshSize,
			tempBuffer.d_pointer(), tempBuffer.sizeInBytes,
			outputBuffer.d_pointer(), outputBuffer.sizeInBytes,

			&accelHandle,
			&emitDesc, 1
		),
		"CudaScene",
		"Failed to build acceleration structure."
	);

	CUDA_SYNC_CHECK();

	// Compact
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	accelBuffer.alloc(compactedSize);
	optixCheck(
		optixAccelCompact(
			optixContext,
			0, 
			accelHandle, 
			accelBuffer.d_pointer(), accelBuffer.sizeInBytes, 
			&accelHandle
		), 
		"CudaScene",
		"Failed to compact acceleration structure."
	);
	cudaSyncCheck("CudaScene", "Failed to synchronize after acceleration structure build.");

	// Clean up
	outputBuffer.free();
	tempBuffer.free();
	compactedSizeBuffer.free();
}

void OptixScene::buildTextures() {
	int numTextures = (int) model->textures.size();

	textureArrays.resize(numTextures);
	textureObjects.resize(numTextures);

	for (int textureId = 0; textureId < numTextures; textureId++) {
		auto texture = model->textures[textureId];

		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc;
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