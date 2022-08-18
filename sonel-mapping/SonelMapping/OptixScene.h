#pragma once
#include "../Cuda/CudaBuffer.h"
#include "Models/SmSbtData.h"
#include "Models/Model.h"
#include "Models/Sonel.h"
#include "Models/SimpelSoundSource.h"


class OptixScene {
public:
	OptixScene(const OptixDeviceContext& optixContext);
	~OptixScene();

	void destroy();
	void clear();
	void build();
	void buildGeometry();
	void buildSonels();
	void buildSoundSources();

	uint32_t getSonelSize() const;
	uint32_t getSoundSourceSize() const;

	void setModel(Model* newModel);
	Model* getModel() const;

	void setSonels(std::vector<Sonel>* newSonels, float searchRadius);
	std::vector<Sonel>* getSonels() const;

	void setSoundSources(std::vector<SimpleSoundSource>* newSoundSources);
	std::vector<SimpleSoundSource>* getSoundSources() const;

	CUdeviceptr getSonelDevicePointer(uint32_t sonelIndex) const;
	CUdeviceptr getSoundSourceDevicePointer(uint32_t soundSourceIndex) const;

	const OptixTraversableHandle& getGeometryHandle() const;
	const OptixTraversableHandle& getSonelHandle() const;
	const OptixTraversableHandle& getInstanceHandle() const;
	
	void fill(uint32_t meshIndex, SmSbtData& triangleData) const;

protected:
	void clearGeometryBuffers();
	void clearSonelBuffers();
	void clearSoundSourceBuffers();

	void prepareGeometryBuffers();
	void prepareSonelBuffers();
	void prepareSoundSourceBuffers();

	void buildGeometryInputs();
	void buildGeometryTextures();
	void buildSonelInputs();
	void buildSoundSourceInputs();
	void buildGeometryAccelStructure();
	void buildSonelAccelStructure();
	void buildSoundSourceAccelStructure();
	void buildInstanceAccelStructure();

	OptixTraversableHandle buildTraversable(const std::vector<OptixBuildInput>& buildInputs, CudaBuffer& accelBuffer);
	static OptixTraversableHandle buildTraversable(const OptixDeviceContext& optixContext, const std::vector<OptixBuildInput>& buildInputs, CudaBuffer& accelBuffer);
	template<class T>
	static CUdeviceptr getDevicePtr(const CudaBuffer& buffer, uint32_t index) {
		return buffer.getCuDevicePointer() + (index * sizeof(T));
	}

	template<class T>
	static void buildAabb(
			const std::vector<T>* aabbItems, CudaBuffer& itemBuffer, CudaBuffer& aabbBuffer,
			std::vector<CUdeviceptr>& cudaPointers, std::vector<OptixBuildInput>& optixInputs, std::vector<uint32_t>& optixInputFlags
	) {
		const auto aabbItemSize = static_cast<uint32_t>(aabbItems->size());
		if (aabbItemSize == 0) {
			return;
		}

		itemBuffer.allocAndUpload(*aabbItems);
		aabbBuffer.alloc(sizeof(OptixAabb) * aabbItemSize);
		CUdeviceptr aabbDevicePtr = aabbBuffer.getCuDevicePointer();

		std::vector<OptixAabb> tempAabbs;
		tempAabbs.resize(aabbItemSize);

		for (uint32_t itemId = 0; itemId < aabbItemSize; itemId++) {
			// upload the model to the device: the builder
			const T& item = (*aabbItems)[itemId];
			const AabbItem& aabbItem = (*aabbItems)[itemId];
			const float radius = aabbItem.getRadius();
			gdt::vec3f position = aabbItem.getPosition();

			if(radius > 1.0f) {
				printf("Constructing AABB with radius %f\n", radius);
				printf("Position %f, %f, %f\n", position.x, position.y, position.z);
			}

			gdt::vec3f min = position - aabbItem.getRadius();
			gdt::vec3f max = position + aabbItem.getRadius();
			tempAabbs[itemId] = {
					min.x,
					min.y,
					min.z,
					max.x,
					max.y,
					max.z
			};

			cudaPointers[itemId] = aabbDevicePtr + (sizeof(OptixAabb) * itemId);

			optixInputs[itemId] = {};
			optixInputs[itemId].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

			if (cudaPointers[itemId] % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT > 0) {
				printf("[OptixScene] Misaligned\n");
			}

			optixInputs[itemId].customPrimitiveArray.aabbBuffers = &(cudaPointers[itemId]);
			optixInputs[itemId].customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
			optixInputs[itemId].customPrimitiveArray.numPrimitives = 1;

			optixInputFlags[itemId] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;

			optixInputs[itemId].customPrimitiveArray.flags = &optixInputFlags[itemId];
			optixInputs[itemId].customPrimitiveArray.numSbtRecords = 1;
			optixInputs[itemId].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
			optixInputs[itemId].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
			optixInputs[itemId].customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;
		}

		aabbBuffer.upload(tempAabbs.data(), aabbItemSize);
	}

protected:
	const OptixDeviceContext& optixContext;
	OptixTraversableHandle geometryHandle;
	OptixTraversableHandle sonelHandle;
	OptixTraversableHandle soundSourceHandle;
	OptixTraversableHandle instanceHandle;

	CUdeviceptr optixInstanceBuffer;
	std::vector<OptixInstance> optixInstances;

	// Model that we will load
	Model* model;
	std::vector<Sonel>* sonels;
	std::vector<SimpleSoundSource>* soundSources;

	uint32_t meshSize;
	uint32_t sonelSize;
	uint32_t soundSourceSize;
	uint32_t instanceSize = 3;

	// Buffer for the accel structure
	CudaBuffer meshAccelBuffer;
	CudaBuffer sonelAccelBuffer;
	CudaBuffer soundSourceAccelBuffer;
	CudaBuffer instanceAccelBuffer;

	// One buffer per mesh
	std::vector<CudaBuffer> vertexBuffer;
	std::vector<CudaBuffer> normalBuffer;
	std::vector<CudaBuffer> texcoordBuffer;
	std::vector<CudaBuffer> indexBuffer;
	
	// Sonel data
	CudaBuffer sonelAabbBuffer;
	CudaBuffer sonelBuffer;
	float sonelRadius = 0.5f;

	CudaBuffer soundSourceAabbBuffer;
	CudaBuffer soundSourceBuffer;

	// Build data
	std::vector<OptixBuildInput> geometryInputs;
	std::vector<uint32_t> geometryInputFlags;
	std::vector<CUdeviceptr> cudaGeometryVertices;
	std::vector<CUdeviceptr> cudaGeometryIndices;
	bool geometryBuffersInvalid = false;

	std::vector<OptixBuildInput> sonelInputs;
	std::vector<uint32_t> sonelInputFlags;
	std::vector<CUdeviceptr> cudaSonelInputs;
	bool sonelBuffersInvalid = false;

	std::vector<OptixBuildInput> soundSourceInputs;
	std::vector<uint32_t> soundSourceInputFlags;
	std::vector<CUdeviceptr> cudaSoundSourceInputs;
	bool soundSourceBuffersInvalid = false;

	// Texture data
	std::vector<cudaArray_t> textureArrays;
	std::vector<cudaTextureObject_t> textureObjects;
};

