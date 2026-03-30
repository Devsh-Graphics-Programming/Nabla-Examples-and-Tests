#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/path_tracing/gaussian_filter.hlsl"

#include "renderer/shaders/pathtrace/common.hlsl"

#include "nbl/examples/common/KeyedQuantizedSequence.hlsl"


using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace nbl::hlsl::path_tracing;

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

    // basics
    uint32_t sampleCount = gSampleCount[launchID]*pc.sensorDynamics.keepAccumulating;

    //
    const float32_t2 pixelSizeNDC = float32_t2(2.f, 2.f) / float32_t2(spirv::LaunchSizeKHR.xy);
    const float32_t2 NDC = float32_t2(launchID.xy) * pixelSizeNDC - promote<float32_t2>(1.f);

    //
    using randgen_type = examples::KeyedQuantizedSequence<Xoroshiro64Star>;
    randgen_type randgen;
    randgen.pSampleBuffer = gScene.init.pSampleSequence;
    randgen.rng = Xoroshiro64Star::construct(gScrambleKey[uint32_t3(launchID.xy & 511,0)]);
    randgen.sequenceSamplesLog2 = gScene.init.sequenceSamplesLog2;

    // take just one sample per dispatch
    float32_t3 albedo, normal;
    {
        float32_t3 randVec = randgen(0u, sampleCount);
        // stochastic reconstruction filter
        const float32_t3 adjNDC = float32_t3(NDC + GaussianFilter<float>::create(1.f,1.f).sample(randVec.xy)*pixelSizeNDC, -1.f);

        // unproject
        const float32_t3 direction = hlsl::normalize(float32_t3(hlsl::mul(pc.sensorDynamics.ndcToRay, adjNDC), -1.0));
        const float32_t3 origin = -float32_t3(direction.xy/direction.z, pc.sensorDynamics.nearClip); // this feels off

        // TODO: remove? do straight to intrinsic?
        RayDesc rayDesc;
        rayDesc.Origin = math::linalg::promoted_mul(pc.sensorDynamics.invView,origin);
        rayDesc.Direction = hlsl::normalize(hlsl::mul(math::linalg::truncate<3,3,3,4>(pc.sensorDynamics.invView),direction));
        rayDesc.TMin = pc.sensorDynamics.nearClip;
        rayDesc.TMax = pc.sensorDynamics.tMax;

        [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]]
        DebugPayload payload;
        payload.albedo = float32_t3(0,0,0);
        payload.worldNormal = float32_t3(0,0,0);
        spirv::traceRayKHR(gTLASes[0], spv::RayFlagsMaskNone, 0xff, 0u, 0u, 0u, rayDesc.Origin, rayDesc.TMin, rayDesc.Direction, rayDesc.TMax, payload);

        albedo = payload.albedo;
        normal = payload.worldNormal;
    }

    gSampleCount[launchID] = ++sampleCount;
    const float32_t rcpSampleCount = 1.f / float32_t(sampleCount);
    //
    float32_t3 prev_albedo = float32_t3(0,0,0);
    float32_t3 prev_normal = float32_t3(0,0,0);
    // read previous frame
    if (rcpSampleCount<1.f)
    {
        prev_albedo = gAlbedo[launchID];
        prev_normal = gNormal[launchID];
    }
    // store albedo
    float32_t3 delta_albedo = (albedo - prev_albedo) * rcpSampleCount;
    if (hlsl::any(hlsl::abs(delta_albedo) > hlsl::promote<float32_t3>(1.0/1023.0)))
        gAlbedo[launchID] = float32_t4(prev_albedo + delta_albedo, 1.0);
    // get it so that -1.0 maps to -511 (513 unsigned so 0.501466275) and 1.0 maps to 511 (0.4995112) and 0 maps to 0
    normal = hlsl::mix(normal*0.499512+promote<float32_t3>(0.999022),normal*0.499512,promote<float32_t3>(0.f)<normal);
    // store normal
    float32_t3 delta_normal = (normal - prev_normal) * rcpSampleCount;
    if (hlsl::any(hlsl::abs(delta_normal) > hlsl::promote<float32_t3>(1.0/1023.0)))
        gNormal[launchID] = float32_t4(prev_normal + delta_normal, 1.0);
    
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
    const float32_t3 geometricNormal = hlsl::cross(vertex1 - vertex0, vertex2 - vertex0);

    const float32_t3x3 normalMatrix = math::linalg::truncate<3,3,3,4>(hlsl::transpose(spirv::WorldToObjectKHR));
    const float32_t3 worldNormal = hlsl::normalize(hlsl::mul(normalMatrix,geometricNormal));

    payload.instanceID = instanceCustomIndex;// TODO: can we get geometry count in instance and "linearize" our geometry into an UUID ?
    payload.primitiveID = primID;

    payload.albedo = float32_t3(1,1,1);
    payload.worldNormal = worldNormal;
}

[shader("miss")]
void miss(inout DebugPayload payload)
{
    payload.albedo = float32_t3(0,0,0); // TODO: sample envmap
    payload.worldNormal = -normalize(spirv::WorldRayDirectionKHR);
}