//
// Created by brent on 24/03/2022.
//

#ifndef SONEL_MAPPING_CUDADEVICEHELPER_H
#define SONEL_MAPPING_CUDADEVICEHELPER_H


inline __device__ float getSoundSourceHitT(const SimpleSoundSource& soundSource, float radius, const gdt::vec3f& rayOrigin, const gdt::vec3f& rayDirection) {
    gdt::vec3f center = soundSource.position;
    gdt::vec3f oc = rayOrigin - center;

    float a = dot(rayDirection, rayDirection);
    float b = 2.0 * dot(oc, rayDirection);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if(discriminant < 0){
        return -1.0;
    }
    else {
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }
}

inline __device__ gdt::vec3f getSoundSourceHit(const SimpleSoundSource& soundSource, float radius, const gdt::vec3f& rayOrigin, const gdt::vec3f& rayDirection) {
    float t = getSoundSourceHitT(soundSource, radius, rayOrigin, rayDirection);
    gdt::vec3f hitPosition = gdt::vec3f();

    if (t > 0.0f) {
        hitPosition = rayOrigin + (t * rayDirection);
    }

    return hitPosition;
};

inline __device__ void getSurfaceData(const SmSbtData& sbtData, gdt::vec3f& hitPosition, gdt::vec3f& geometryNormal, gdt::vec3f& shadingNormal) {
    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int   primitiveIndex = optixGetPrimitiveIndex();

    const gdt::vec3i index = sbtData.index[primitiveIndex];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    hitPosition = (1.f - u - v) * sbtData.vertex[index.x]
                  + u * sbtData.vertex[index.y]
                  + v * sbtData.vertex[index.z];

    const gdt::vec3f& A = sbtData.vertex[index.x];
    const gdt::vec3f& B = sbtData.vertex[index.y];
    const gdt::vec3f& C = sbtData.vertex[index.z];
    geometryNormal = normalize(gdt::cross(B - A, C - A));
    shadingNormal = (sbtData.normal)
                    ? ((1.f - u - v) * sbtData.normal[index.x]
                       + u * sbtData.normal[index.y]
                       + v * sbtData.normal[index.z])
                    : geometryNormal;

    shadingNormal = normalize(shadingNormal);
}

inline __device__ void fixNormals(const gdt::vec3f& rayDirection, gdt::vec3f& geometryNormal, gdt::vec3f& shadingNormal) {
    if(dot(rayDirection, geometryNormal) > 0.f)
        geometryNormal = -geometryNormal;
    geometryNormal = normalize(geometryNormal);

    if(dot(geometryNormal, shadingNormal) < 0.f)
        shadingNormal -= 2.f * dot(geometryNormal, shadingNormal) * geometryNormal;
    shadingNormal = normalize(shadingNormal);
}

#endif //SONEL_MAPPING_CUDADEVICEHELPER_H
