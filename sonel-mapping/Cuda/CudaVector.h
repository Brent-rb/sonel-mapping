//
// Created by brent on 26/01/2022.
//

#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>

#ifndef SONEL_MAPPING_CUDAVECTOR_H
#define SONEL_MAPPING_CUDAVECTOR_H

template <typename T>
class CudaVector {
public:
    __device__ CudaVector(): data(nullptr), index(0), dataSize(0) {
        alloc(32);
    };

    __device__ ~CudaVector() {
        if (data != nullptr) {
            free(data);
        }
    }

    __device__ void push(const T& element) {
        if (index == dataSize) {
            alloc(dataSize * 2);
        }

        data[index] = element;
        index++;
    }

    __device__ T& pop() {
        T& element = data[index - 1];
        index--;

        return element;
    }

    __device__ uint64_t size() const {
        return dataSize;
    }

protected:
    __device__ void alloc(const uint64_t newSize) {
        T* newData = reinterpret_cast<T*>(malloc(newSize * sizeof(T)));

        if (index > 0 && data != nullptr) {
            int copySize = index;
            if (newSize < copySize) {
                copySize = newSize;
            }

            memcpy(newData, data, copySize * sizeof(T));
        }

        if (data != nullptr)
            free(data);

        data = newData;
        dataSize = newSize;
    }

private:
    T* data;
    uint64_t index;
    uint64_t dataSize;
};


#endif //SONEL_MAPPING_CUDAVECTOR_H
