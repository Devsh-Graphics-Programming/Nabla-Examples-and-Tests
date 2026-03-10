#include "renderer/shaders/pathtrace/common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"

using namespace nbl;
using namespace hlsl;
using namespace nbl::this_example;

[[vk::push_constant]] SDebugPushConstants pc;


struct [raypayload] DebugPayload
{
    uint32_t instanceID : read(caller) : write(closesthit);
    uint32_t primitiveID : read(caller) : write(closesthit);
    float32_t3 albedo : read(caller) : write(closesthit,miss);
    float32_t3 worldNormal : read(caller) : write(closesthit,miss);
};

[shader("raygeneration")]
void raygen()
{
    const uint32_t3 launchID = spirv::LaunchIdKHR;
    const uint32_t3 launchSize = spirv::LaunchSizeKHR;

    float32_t2 coord = (float32_t3(launchID) / float32_t3(launchSize)).xy;
    // float32_t3 NDC = float32_t3(coord * 2.0 - 1.0, 1.0);
    float32_t2 NDC = float32_t2(coord * 2.0 - 1.0);
    float32_t3 direction = normalize(float32_t3(mul(pc.sensorDynamics.ndcToRay, NDC), -1.0));
    float32_t3 origin = -float32_t3(direction.xy/direction.z, pc.sensorDynamics.nearClip);

    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = pc.sensorDynamics.nearClip;
    rayDesc.TMax = pc.sensorDynamics.tMax;
    
    [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]]
    DebugPayload payload;
    payload.albedo = float32_t3(0,0,0);
    // payload.pcg = PrimaryPayload::generator_t::construct(rnd());
    spirv::traceRayKHR(gTLASes[0], spv::RayFlagsMaskNone, 0xff, 0u, 0u, 0u, rayDesc.Origin, rayDesc.TMin, rayDesc.Direction, rayDesc.TMax, payload);

    gAlbedo[launchID] = float32_t4(payload.albedo, 1.0);
    // gAlbedo[launchID] = float32_t4(float32_t3(launchID)/float32_t3(launchSize),1.f);
}

[shader("closesthit")]
void closesthit(inout DebugPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int primID = spirv::PrimitiveId;
    const int instanceCustomIndex = spirv::InstanceCustomIndexKHR;
    const int geometryIndex = spirv::RayGeometryIndexKHR;

    // const float32_t3 vertexNormal = calculateNormals(primID, geom, attribs.barycentrics);
    // const float32_t3 worldNormal = normalize(mul(vertexNormal, transpose(spirv::WorldToObjectKHR)).xyz);

    payload.instanceID = instanceCustomIndex;
    payload.primitiveID = primID;

    payload.albedo = float32_t3(1,1,1);
    // payload.worldNormal = worldNormal;
}

[shader("miss")]
void miss(inout DebugPayload payload)
{
    payload.albedo = float32_t3(0.1,0.1,0.1);
    payload.worldNormal = float32_t3(0,0,0);
}