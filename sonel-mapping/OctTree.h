#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

template <typename T>
struct OctTreeItem {
	T data;
	gdt::vec3f position;
};

template <typename T>
struct OctTreeResult {
	OctTreeItem<T>* data;
	uint32_t maxItems;
	uint32_t currentItems;
};

class BoundingBox {
public:
	BoundingBox(): lowerBound(gdt::vec3f()), upperBound(gdt::vec3f(1.0f)) {
	}

	BoundingBox(gdt::vec3f lowerBound, gdt::vec3f upperBound) : lowerBound(lowerBound), upperBound(upperBound) {

	}

	~BoundingBox() {
	
	}

	float getWidth() {
		return upperBound.x - lowerBound.x;
	}

	float getDepth() {
		return upperBound.z - lowerBound.z;
	}

	float getHeight() {
		return upperBound.y - lowerBound.y;
	}

	gdt::vec3f getDimensions() {
		return upperBound - lowerBound;
	}

	__device__ __host__ bool contains(const gdt::vec3f& point) {
		return
			(point.x >= lowerBound.x && point.x <= upperBound.x) &&
			(point.y >= lowerBound.y && point.y <= upperBound.y) &&
			(point.z >= lowerBound.z && point.z <= upperBound.z);
	}

	gdt::vec3f lowerBound;
	gdt::vec3f upperBound;
};

template <typename T>
class OctTree {
public:
	OctTree() : data(nullptr), children(nullptr), maxItems(0), currentItems(0) {

	}

	OctTree(const BoundingBox& boundingBox, const uint32_t& maxItems) : OctTree() {
		init(boundingBox, maxItems);
	}

	void destroy() {
		if (children != nullptr) {
			delete[] children;
		}
		if (data != nullptr) {
			delete[] data;
		}
	}

	void init(const BoundingBox& boundingBox, const uint32_t& maxItems) {
		// Init may only be called in the beginning
		assert(children == nullptr && data == nullptr);

		this->boundingBox = boundingBox;
		this->maxItems = maxItems;
		this->currentItems = 0;

		data = new OctTreeItem<T>[maxItems];
		children = nullptr;
	}

	bool insert(const T* item, const gdt::vec3f& position) {
		/*printf("Inserting [%.2f, %.2f, %.2f] into [%.2f -> %.2f, %.2f -> %.2f, %.2f -> %.2f], current: %d, max: %d\n",
			position.x, position.y, position.z,
			boundingBox.lowerBound.x, boundingBox.upperBound.x,
			boundingBox.lowerBound.y, boundingBox.upperBound.y,
			boundingBox.lowerBound.z, boundingBox.upperBound.z,
			currentItems, maxItems
		);*/

		if (!boundingBox.contains(position)) {
			// printf("OctTree does not contain position, not inserting\n");
			return false;
		}

		if (isLeaf()) {
			if (hasRoom()) {
				// printf("OctTree contains position, inserting in data\n");
				insertInHere(item, position);
				return true;
			}
			else {
				// printf("OctTree contains position, data is full -> splitting octtree.\n");
				split();
				return insertInChildren(item, position);
			}
		}
		else {
			// printf("OctTree contains position, inserting in children.\n");
			return insertInChildren(item, position);
		}
	}

	__device__ __host__ OctTreeResult<T> get(const gdt::vec3f& point) {
		if (!contains(point)) {
			return {
				nullptr,	// Data
				0,		// maxItems
				0		// currentItems
			};
		}
		
		OctTree* children = this->children;
		int index = 0;

		while (index < 8 && children != nullptr) {
			// printf("Checking child %d\n", index);
			OctTree& octTree = children[index];

			if (octTree.contains(point)) {
				// printf("\tContains point\n");
				if (octTree.isLeaf()) {
					// printf("\Is leaf\n");
					return {
						octTree.data,
						octTree.maxItems,
						octTree.currentItems
					};
				}
				else {
					// printf("\Is node, going deeper...\n");
					children = octTree.children;
					index = 0;
				}
			}
			else {
				index++;
			}
		}

		return {
				nullptr,	// Data
				0,		// maxItems
				0		// currentItems
		};
	}

	uint32_t getCount() {
		return currentItems;
	}

	uint32_t getMaxSize() {
		return maxItems;
	}

	OctTree<T>* upload() {
		OctTree<T>* deviceOctTree;

		// Allocate space for base class on CUDA Device
		cudaMalloc((void**)&deviceOctTree, sizeof(OctTree<T>));
		// Copy data from host to cuda device
		cudaMemcpy(deviceOctTree, this, sizeof(OctTree<T>), cudaMemcpyHostToDevice);

		upload(deviceOctTree);

		return deviceOctTree;
	}

	void upload(OctTree<T>* deviceOctTree) {
		if (isLeaf()) {
			OctTreeItem<T>* octTreeData;

			// Allocate space for the data array
			CUDA_CHECK(cudaMalloc((void**)&octTreeData, maxItems * sizeof(OctTreeItem<T>)));
			// Copy data
			CUDA_CHECK(cudaMemcpy(octTreeData, data, currentItems * sizeof(OctTreeItem<T>), cudaMemcpyHostToDevice));

			// Copy pointer over
			CUDA_CHECK(cudaMemcpy(&(deviceOctTree->data), &octTreeData, sizeof(OctTreeItem<T>*), cudaMemcpyHostToDevice));
		}
		else {
			OctTree<T>* octTreeChildren;

			// Allocate space for the children array
			CUDA_CHECK(cudaMalloc((void**)&octTreeChildren, 8 * sizeof(OctTree<T>)));
			// Copy children
			CUDA_CHECK(cudaMemcpy(octTreeChildren, children, 8 * sizeof(OctTree<T>), cudaMemcpyHostToDevice));

			// Copy pointer over
			CUDA_CHECK(cudaMemcpy(&(deviceOctTree->children), &octTreeChildren, sizeof(OctTree<T>*), cudaMemcpyHostToDevice));
			

			for (int i = 0; i < 8; i++) {
				children[i].upload(&octTreeChildren[i]);
			}
		}
	}

	static void clear(OctTree<T>* devicePtr) {
		OctTree<T> octTree;
		CUDA_CHECK(cudaMemcpy(&octTree, devicePtr, sizeof(OctTree<T>), cudaMemcpyDeviceToHost));

		if (octTree.children != nullptr) {
			for (int i = 0; i < 8; i++) {
				clear(&(octTree.children[i]));
			}
			CUDA_CHECK(cudaFree(octTree.children));
		}
		CUDA_CHECK(cudaFree(octTree.data));
	}

protected:
	__device__ __host__ bool contains(const gdt::vec3f& point) {
		return boundingBox.contains(point);
	}

	__device__ __host__ bool isLeaf() {
		return children == nullptr;
	}

	bool hasRoom() {
		return currentItems < maxItems;
	}

	void split() {
		gdt::vec3f dimensions = boundingBox.getDimensions();
		gdt::vec3f halfDimensions = dimensions / 2.0f;
		gdt::vec3f lowerBound = boundingBox.lowerBound;

		gdt::vec3f lb1 = lowerBound;
		gdt::vec3f lb2 = lowerBound + gdt::vec3f(halfDimensions.x, 0.0f, 0.0f);
		gdt::vec3f lb3 = lowerBound + gdt::vec3f(0.0f, 0.0f, halfDimensions.z);
		gdt::vec3f lb4 = lowerBound + gdt::vec3f(halfDimensions.x, 0.0f, halfDimensions.z);
		gdt::vec3f lb5 = lb1 + gdt::vec3f(0.0f, halfDimensions.y, 0.0f);
		gdt::vec3f lb6 = lb2 + gdt::vec3f(0.0f, halfDimensions.y, 0.0f);
		gdt::vec3f lb7 = lb3 + gdt::vec3f(0.0f, halfDimensions.y, 0.0f);
		gdt::vec3f lb8 = lb4 + gdt::vec3f(0.0f, halfDimensions.y, 0.0f);

		gdt::vec3f ub1 = lb1 + halfDimensions;
		gdt::vec3f ub2 = lb2 + halfDimensions;
		gdt::vec3f ub3 = lb3 + halfDimensions;
		gdt::vec3f ub4 = lb4 + halfDimensions;
		gdt::vec3f ub5 = lb5 + halfDimensions;
		gdt::vec3f ub6 = lb6 + halfDimensions;
		gdt::vec3f ub7 = lb7 + halfDimensions;
		gdt::vec3f ub8 = lb8 + halfDimensions;

		BoundingBox bb1 = BoundingBox(lb1, ub1);
		BoundingBox bb2 = BoundingBox(lb2, ub2);
		BoundingBox bb3 = BoundingBox(lb3, ub3);
		BoundingBox bb4 = BoundingBox(lb4, ub4);
		BoundingBox bb5 = BoundingBox(lb5, ub5);
		BoundingBox bb6 = BoundingBox(lb6, ub6);
		BoundingBox bb7 = BoundingBox(lb7, ub7);
		BoundingBox bb8 = BoundingBox(lb8, ub8);

		children = new OctTree[8];
		children[0].init(bb1, maxItems);
		children[1].init(bb2, maxItems);
		children[2].init(bb3, maxItems);
		children[3].init(bb4, maxItems);
		children[4].init(bb5, maxItems);
		children[5].init(bb6, maxItems);
		children[6].init(bb7, maxItems);
		children[7].init(bb8, maxItems);

		for (int i = 0; i < currentItems; i++) {
			OctTreeItem<T>& item = data[i];
			assert(insertInChildren(&(item.data), item.position));
		}

		delete[] data;
		data = nullptr;
		currentItems = 0;
	}

	void insertInHere(const T* item, const gdt::vec3f& position) {
		// Make sure that we have no out of bounds error
		assert(currentItems < maxItems);

		data[currentItems] = {
			*item,
			position
		};

		currentItems++;
	}

	bool insertInChildren(const T* item, const gdt::vec3f& position) {
		// Make sure we have child nodes first
		assert(children != nullptr);

		bool inserted = false;
		for (int i = 0; i < 8; i++) {
			if (children[i].insert(item, position)) {
				inserted = true;
				break;
			}
		}

		return inserted;
	}

protected:
	BoundingBox boundingBox;
	OctTree* children;
	OctTreeItem<T>* data;
	uint32_t maxItems;
	uint32_t currentItems;
};

