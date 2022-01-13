#pragma once
#include "CUDABuffer.h"
#include "TriangleMeshSbtData.h"
#include "Model.h"

class OptixScene {
public:
	OptixScene(const OptixDeviceContext& optixContext, const Model* model);
	~OptixScene();

	const OptixTraversableHandle& getTraversableHandle() const;
	const Model* getModel() const;
	void fill(const uint32_t meshIndex, TriangleMeshSbtData& triangleData) const;

protected:
	void build(const OptixDeviceContext& optixContext);

	void prepareBuffers(
		std::vector<OptixBuildInput>& triangleInputs,
		std::vector<uint32_t>& triangleInputFlags,
		std::vector<CUdeviceptr>& cudaVertices,
		std::vector<CUdeviceptr>& cudaIndices
	);

	void buildTriangleInput(
		std::vector<OptixBuildInput>& triangleInputs, 
		std::vector<uint32_t>& triangleInputFlags, 
		std::vector<CUdeviceptr>& cudaVertices, 
		std::vector<CUdeviceptr>& cudaIndices
	);

	void buildAccelStructure(
		const OptixDeviceContext& optixContext,
		const std::vector<OptixBuildInput>& triangleInputs,
		OptixTraversableHandle& accelHandle
	);

	void buildTextures();

protected:
	// Model that we will load
	const Model* model;
	uint32_t meshSize;

	OptixTraversableHandle traversableHandle;
	// Buffer for the accell structure
	CUDABuffer accelBuffer;


	// One buffer per mesh
	std::vector<CUDABuffer> vertexBuffer;
	std::vector<CUDABuffer> normalBuffer;
	std::vector<CUDABuffer> texcoordBuffer;
	std::vector<CUDABuffer> indexBuffer;


	// Texture data
	std::vector<cudaArray_t> textureArrays;
	std::vector<cudaTextureObject_t> textureObjects;
};

