#pragma once
#include "../Cuda/CudaBuffer.h"
#include "Models/TriangleMeshSbtData.h"
#include "Models/Model.h"
#include "Models/Sonel.h"

class OptixScene {
public:
	OptixScene(const OptixDeviceContext& optixContext);
	~OptixScene();

	void clear();
	void build();
	void buildTriangles();
	void buildSonels();

	uint32_t getSonelSize() const;

	void setModel(Model* newModel);
	Model* getModel() const;

	void setSonels(std::vector<Sonel>* newSonels, float searchRadius);
	std::vector<Sonel>* getSonels() const;
	CUdeviceptr getSonelDevicePointer(uint32_t sonelIndex) const;

	const OptixTraversableHandle& getGeoTraversable() const;
	const OptixTraversableHandle& getAabbTraversable() const;
	const OptixTraversableHandle& getInstanceTraversable() const;
	
	void fill(const uint32_t meshIndex, TriangleMeshSbtData& triangleData) const;

protected:
	void clearMeshBuffers();
	void clearSonelBuffers();

	void prepareTriangleBuffers();
	void prepareSonelBuffers();

	void buildTriangleInputs();
	void buildTextures();
	void buildSonelAabbInputs();
	void buildTriangleAccelStructure();
	void buildSonelAabbAccelStructure();
	void buildInstanceStructure();

	OptixTraversableHandle buildTraversable(const std::vector<OptixBuildInput>& buildInputs, CudaBuffer& accelBuffer);
	static OptixTraversableHandle buildTraversable(const OptixDeviceContext& optixContext, const std::vector<OptixBuildInput>& buildInputs, CudaBuffer& accelBuffer);

protected:
	const OptixDeviceContext& optixContext;
	OptixTraversableHandle triangleHandle;
	OptixTraversableHandle aabbHandle;
	OptixTraversableHandle instanceHandle;

	CUdeviceptr optixInstanceBuffer;
	std::vector<OptixInstance> optixInstances;


	// Model that we will load
	Model* model;
	std::vector<Sonel>* sonels;

	uint32_t meshSize;
	uint32_t sonelSize;

	
	// Buffer for the accell structure
	CudaBuffer triangleAccelBuffer;
	CudaBuffer aabbAccelBuffer;
	CudaBuffer instanceAccelBuffer;

	// One buffer per mesh
	std::vector<CudaBuffer> vertexBuffer;
	std::vector<CudaBuffer> normalBuffer;
	std::vector<CudaBuffer> texcoordBuffer;
	std::vector<CudaBuffer> indexBuffer;
	
	// Sonel data
	CudaBuffer sonelAabbBuffer;
	CudaBuffer sonelBuffer;
	float radius = 0.5f;

	// Build data
	std::vector<OptixBuildInput> triangleInputs;
	std::vector<uint32_t> triangleInputFlags;
	std::vector<CUdeviceptr> cudaVertices;
	std::vector<CUdeviceptr> cudaIndices;
	bool meshBuffersInvalid = false;

	std::vector<OptixBuildInput> aabbInputs;
	std::vector<uint32_t> aabbInputFlags;
	std::vector<CUdeviceptr> cudaAabbs;
	bool sonelBuffersInvalid = false;

	// Texture data
	std::vector<cudaArray_t> textureArrays;
	std::vector<cudaTextureObject_t> textureObjects;
};

