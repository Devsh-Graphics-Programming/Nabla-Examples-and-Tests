#include "renderer/shaders/pathtrace/common.hlsl"
#include "renderer/shaders/pathtrace/rand_gen.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/path_tracing/gaussian_filter.hlsl"

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

    uint2 scrambleDim;
    gScrambleKey.GetDimensions(scrambleDim.x, scrambleDim.y);
    float32_t2 pixOffsetParam = (float32_t2)1.0 / float32_t2(scrambleDim);

    float32_t2 coord = (float32_t3(launchID) / float32_t3(launchSize)).xy;
    uint32_t2 texCoord = uint32_t2(launchID.x & 511, launchID.y & 511);
    using randgen_type = RandomUniformND<Xoroshiro64Star,3>;
    randgen_type randgen = randgen_type::create(gScrambleKey[texCoord], pc.sensorDynamics.pSampleSequence);
    float32_t3 NDC = float32_t3(coord * 2.0 - 1.0, -1.0);

    float32_t3 acc_albedo = float32_t3(0,0,0);
    float32_t3 acc_normal = float32_t3(0,0,0);
    uint32_t sampleCount = pc.sensorDynamics.maxSPP;
    float rcpSampleCount = 1.0 / float(sampleCount);
    for (uint32_t i = 0; i < sampleCount; i++)
    {
        float32_t3 randVec = randgen(0u, i);
        path_tracing::GaussianFilter<float> filter = path_tracing::GaussianFilter<float>::create(2.5, 1.5); // stochastic reconstruction filter
        float32_t3 adjNDC = NDC;
        adjNDC.xy += pixOffsetParam * filter.sample(randVec.xy);
        float32_t3 direction = hlsl::normalize(float32_t3(hlsl::mul(pc.sensorDynamics.ndcToRay, adjNDC), -1.0));
        float32_t3 origin = -float32_t3(direction.xy/direction.z, pc.sensorDynamics.nearClip);

        RayDesc rayDesc;
        rayDesc.Origin = math::linalg::promoted_mul(pc.sensorDynamics.invView, origin);
        rayDesc.Direction = hlsl::normalize(hlsl::mul(math::linalg::truncate<3,3,3,4>(pc.sensorDynamics.invView), direction));
        rayDesc.TMin = pc.sensorDynamics.nearClip;
        rayDesc.TMax = pc.sensorDynamics.tMax;

        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]]
        DebugPayload payload;
        payload.albedo = float32_t3(0,0,0);
        payload.worldNormal = float32_t3(0,0,0);
        spirv::traceRayKHR(gTLASes[0], spv::RayFlagsMaskNone, 0xff, 0u, 0u, 0u, rayDesc.Origin, rayDesc.TMin, rayDesc.Direction, rayDesc.TMax, payload);

        acc_albedo += payload.albedo;
        acc_normal += payload.worldNormal * 0.5 + 0.5;
    }

    const bool firstFrame = pc.sensorDynamics.rcpFramesDispatched == 1.0;
    // clear accumulations totally if beginning a new frame
    if (firstFrame)
    {
        gAlbedo[launchID] = float32_t4(acc_albedo * rcpSampleCount, 1.0);
        gNormal[launchID] = float32_t4(acc_normal * rcpSampleCount, 1.0);
    }
    else
    {
        float32_t3 prev_albedo = gAlbedo[launchID];
        float32_t3 delta = (acc_albedo * rcpSampleCount - prev_albedo) * pc.sensorDynamics.rcpFramesDispatched;
        if (hlsl::any(delta > hlsl::promote<float32_t3>(1.0/1024.0)))
            gAlbedo[launchID] = float32_t4(prev_albedo + delta, 1.0);

        float32_t3 prev_normal = gNormal[launchID];
        delta = (acc_normal * rcpSampleCount - prev_normal) * pc.sensorDynamics.rcpFramesDispatched;
        if (hlsl::any(delta > hlsl::promote<float32_t3>(1.0/512.0)))
            gNormal[launchID] = float32_t4(prev_normal + delta, 1.0);
    }
    
}

[shader("closesthit")]
void closesthit(inout DebugPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int primID = spirv::PrimitiveId;
    const int instanceCustomIndex = spirv::InstanceCustomIndexKHR;
    const int geometryIndex = spirv::RayGeometryIndexKHR;
    
    float32_t3 vertex0 = spirv::HitTriangleVertexPositionsKHR[0];
    float32_t3 vertex1 = spirv::HitTriangleVertexPositionsKHR[1];
    float32_t3 vertex2 = spirv::HitTriangleVertexPositionsKHR[2];
    const float32_t3 vertexNormal = hlsl::cross(vertex1 - vertex0, vertex2 - vertex0);
    const float32_t3 worldNormal = hlsl::normalize(hlsl::mul(math::linalg::truncate<3,3,3,4>(hlsl::transpose(spirv::ObjectToWorldKHR)), vertexNormal));

    payload.instanceID = instanceCustomIndex;
    payload.primitiveID = primID;

    payload.albedo = float32_t3(1,1,1);
    payload.worldNormal = worldNormal;
}

[shader("miss")]
void miss(inout DebugPayload payload)
{
    payload.albedo = float32_t3(0,0,0);
    payload.worldNormal = -spirv::WorldRayDirectionKHR;
}