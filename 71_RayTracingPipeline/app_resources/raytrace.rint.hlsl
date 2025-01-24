#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

struct Ray
{
    float32_t3 origin;
    float32_t3 direction;
};

struct Attrib
{
    float3 HitAttribute;
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

    if (discriminant < 0)
    {
        return -1.0;
    }
    else
    {
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }
}

[shader("intersection")]
void main()
{
    Ray ray;
    ray.origin = WorldRayOrigin();
    ray.direction = WorldRayDirection();

    const int primID = PrimitiveIndex();

    // Sphere data
    SProceduralGeomInfo sphere = vk::RawBufferLoad < SProceduralGeomInfo > (pc.proceduralGeomInfoBuffer + primID * sizeof(SProceduralGeomInfo));

    float32_t tHit = hitSphere(sphere, ray);
    
    Attrib attrib;
    // Report hit point
    if (tHit > 0)
        ReportHit(tHit, 0, attrib);
}