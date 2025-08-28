#include "common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

struct Ray
{
    float32_t3 origin;
    float32_t3 direction;
};

// Ray-Sphere intersection
// http://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
float32_t hitSphere(SProceduralGeomInfo s, Ray r)
{
    float32_t3 oc = r.origin - s.center;
    float32_t a = dot(r.direction, r.direction);
    float32_t b = 2.0 * dot(oc, r.direction);
    float32_t c = dot(oc, oc) - s.radius * s.radius;
    float32_t discriminant = b * b - 4 * a * c;

    // return whatever, if the discriminant is negative, it will produce a NaN, and NaN will compare false
    return (-b - sqrt(discriminant)) / (2.0 * a);
}

[shader("intersection")]
void main()
{
    Ray ray;
    ray.origin = spirv::WorldRayOriginKHR;
    ray.direction = spirv::WorldRayDirectionKHR;

    const int primID = spirv::PrimitiveId;

    // Sphere data
    SProceduralGeomInfo sphere = vk::RawBufferLoad<SProceduralGeomInfo>(pc.proceduralGeomInfoBuffer + primID * sizeof(SProceduralGeomInfo));

    const float32_t tHit = hitSphere(sphere, ray);
    
    [[vk::ext_storage_class(spv::StorageClassHitAttributeKHR)]]
    ProceduralHitAttribute hitAttrib;

    // Report hit point
    if (tHit > 0)
    {
        hitAttrib.center = sphere.center;
        spirv::reportIntersectionKHR(tHit, 0);
    }
}