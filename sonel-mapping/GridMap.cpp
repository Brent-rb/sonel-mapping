#include "GridMap.h"

GridSonelMap::GridSonelMap() : pages(nullptr), resolution(0.0f), bounds() {

}

GridSonelMap::GridSonelMap(const std::vector<Sonel>& sonels, const gdt::box3f bounds, const float resolution): GridSonelMap() {
	parse(sonels, bounds, resolution);
}

GridSonelMap::~GridSonelMap() {
	if (pages != nullptr) {
		delete[] pages;
		pages = nullptr;
	}
}


void GridSonelMap::parse(const std::vector<Sonel>& sonels, const gdt::box3f bounds, const float resolution) {
	this->bounds = bounds;
	this->resolution = resolution;
	
	if (pages != nullptr) {
		delete[] pages;
		pages = nullptr;
	}

	gdt::vec3f dimensions = bounds.dims / resolution;
	uint64_t width = static_cast<uint64_t>(roundf(dimensions.x));
	uint64_t height = static_cast<uint64_t>(roundf(dimensions.y));
	uint64_t depth = static_cast<uint64_t>(roundf(dimensions.z));
	size = (width + 1) * (height + 1) * (depth + 1);

	std::vector<Sonel>* tempPages = new std::vector<Sonel>[size];
	
	for (int i = 0; i < sonels.size(); i++) {
		const Sonel& sonel = sonels[i];

		gdt::vec3f position = (sonel.position - bounds.lower) / resolution;
		uint64_t width = static_cast<uint64_t>(roundf(position.x));
		uint64_t height = static_cast<uint64_t>(roundf(position.y));
		uint64_t depth = static_cast<uint64_t>(roundf(position.z));
		uint64_t index = width * height * depth;

		tempPages[index].push_back(sonel);
	}

	pages = new GridSonelPage[size];
	for (int i = 0; i < size; i++) {
		std::vector<Sonel>& tempPage = tempPages[i];
		GridSonelPage& page = pages[i];

		if (tempPage.size() == 0) {
			page.size = 0;
			page.sonels = nullptr;
			continue;
		}

		page.size = tempPage.size();
		page.sonels = new Sonel[page.size];

		memcpy(page.sonels, tempPage.data(), page.size * sizeof(Sonel));
	}

	delete[] tempPages;
}

__device__ __host__ GridSonelPage& GridSonelMap::get(gdt::vec3f& point) {
	gdt::vec3f position = (point - bounds.lower) / resolution;
	uint64_t width = static_cast<uint64_t>(roundf(position.x));
	uint64_t height = static_cast<uint64_t>(roundf(position.y));
	uint64_t depth = static_cast<uint64_t>(roundf(position.z));
	uint64_t index = width * height * depth;

	return pages[index];
}

GridSonelMap* GridSonelMap::upload() {
	GridSonelMap* deviceSonelMap;

	// Allocate space for base class on CUDA Device
	cudaMalloc((void**)&deviceSonelMap, sizeof(GridSonelMap));
	// Copy data from host to cuda device
	cudaMemcpy(deviceSonelMap, this, sizeof(GridSonelMap), cudaMemcpyHostToDevice);
	
	GridSonelPage* devicePages;
	
	cudaMalloc((void**)&devicePages, size * sizeof(GridSonelPage));
	cudaMemcpy(devicePages, this->pages, size * sizeof(GridSonelPage), cudaMemcpyHostToDevice);
	cudaMemcpy(&(deviceSonelMap->pages), &devicePages, sizeof(GridSonelPage*), cudaMemcpyHostToDevice);

	for (int i = 0; i < size; i++) {
		GridSonelPage& page = pages[i];

		if (page.sonels != nullptr) {
			// TODO
		}
	}

	return deviceSonelMap;
}