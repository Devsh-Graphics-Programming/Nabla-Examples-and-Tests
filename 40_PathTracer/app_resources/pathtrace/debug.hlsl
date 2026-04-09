#include "common.hlsl"


[[vk::push_constant]] SDebugPushConstants pc;


struct [raypayload] DebugPayload
{
    uint32_t instanceID : read(caller) : write(closesthit);
    uint32_t primitiveID : read(caller) : write(closesthit);
    SArbitraryOutputValues aov : read(caller) : write(closesthit,miss);
};

[shader("raygeneration")]
void raygen()
{
    const uint16_t3 launchID = uint16_t3(spirv::LaunchIdKHR);
    
    SPixelSamplingInfo samplingInfo = advanceSampleCount(launchID,1,uint16_t(pc.sensorDynamics.keepAccumulating));
    // took 64k-1 spp
    if (samplingInfo.rcpNewSampleCount==0.f)
        return;

    //
    const float32_t2 pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(spirv::LaunchSizeKHR.xy);
    const float32_t2 NDC = float32_t2(launchID.xy) * pixelSizeNDC - promote<float32_t2>(1.f);

    [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]]
    DebugPayload payload;
    // take just one sample per dispatch
    {        
        const float16_t2 randVec = float16_t2(samplingInfo.randgen(0u,samplingInfo.firstSample).xy);
        const SRay ray = SRay::create(pc.sensorDynamics,pixelSizeNDC,NDC,randVec);

        payload.aov.clear();
        spirv::traceRayKHR(gTLASes[0], spv::RayFlagsMaskNone, 0xff, 0u, 0u, 0u, ray.origin, ray.tMin, ray.direction, ray.tMax, payload);
    }

    // simple overwrite without accumulation
    gRWMCCascades[launchID] = uint32_t2(payload.instanceID,payload.primitiveID);
    // can also shove some stuff in `gBeauty`, `gMotion` and `gMask`

    const bool keepAccumulating = samplingInfo.firstSample;
    // albedo
    Accumulator<ImageAccessor_gAlbedo> albedoAcc;
    albedoAcc.accumulate(launchID.xy,launchID.z,float32_t3(payload.aov.albedo),samplingInfo.rcpNewSampleCount,keepAccumulating);
    // normal
    Accumulator<ImageAccessor_gNormal> normalAcc;
    normalAcc.accumulate(launchID.xy,launchID.z,float32_t3(correctSNorm10WhenStoringToUnorm(payload.aov.normal)),samplingInfo.rcpNewSampleCount,keepAccumulating);
}

[shader("closesthit")]
void closestHit(inout DebugPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const uint32_t instanceCustomIndex = spirv::InstanceCustomIndexKHR;
    const uint32_t geometryIndex = spirv::RayGeometryIndexKHR;
    payload.instanceID = instanceCustomIndex;// TODO: can we get geometry count in instance and "linearize" our geometry into an UUID ?
    payload.primitiveID = spirv::PrimitiveId;

    payload.aov.albedo = accum_t(1,1,1);
    payload.aov.normal = accum_t(reconstructGeometricNormal());
}

[shader("miss")]
void miss(inout DebugPayload payload)
{
    const SEnvSample _sample = sampleEnv(spirv::WorldRayDirectionKHR);
    //_sample.color;
    payload.aov = _sample.aov;
}