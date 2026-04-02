#include "common.hlsl"


[[vk::push_constant]] SDebugPushConstants pc;


using accum_t = float16_t3;
struct [raypayload] DebugPayload
{
    uint32_t instanceID : read(caller) : write(closesthit);
    uint32_t primitiveID : read(caller) : write(closesthit);
    accum_t albedo : read(caller) : write(closesthit,miss);
    accum_t worldNormal : read(caller) : write(closesthit,miss);
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

        payload.albedo = accum_t(0,0,0);
        payload.worldNormal = accum_t(0,0,0);
        spirv::traceRayKHR(gTLASes[0], spv::RayFlagsMaskNone, 0xff, 0u, 0u, 0u, ray.origin, ray.tMin, ray.direction, ray.tMax, payload);
    }

    // simple overwrite without accumulation
    gRWMCCascades[launchID] = uint32_t2(payload.instanceID,payload.primitiveID);
    // can also shove some stuff in `gBeauty`, `gMotion` and `gMask`

    // albedo
    Accumulator<ImageAccessor_gAlbedo> albedoAcc;
    albedoAcc.accumulate(launchID.xy,launchID.z,float32_t3(payload.albedo),samplingInfo.rcpNewSampleCount);
    // normal
    Accumulator<ImageAccessor_gNormal> normalAcc;
    normalAcc.accumulate(launchID.xy,launchID.z,float32_t3(correctSNorm10WhenStoringToUnorm(payload.worldNormal)),samplingInfo.rcpNewSampleCount);
}

[shader("closesthit")]
void closesthit(inout DebugPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.worldNormal = accum_t(reconstructGeometricNormal());

    const uint32_t instanceCustomIndex = spirv::InstanceCustomIndexKHR;
    const uint32_t geometryIndex = spirv::RayGeometryIndexKHR;
    payload.instanceID = instanceCustomIndex;// TODO: can we get geometry count in instance and "linearize" our geometry into an UUID ?
    payload.primitiveID = spirv::PrimitiveId;

    payload.albedo = accum_t(1,1,1);
}

[shader("miss")]
void miss(inout DebugPayload payload)
{
    const SSpectralType _sample = sampleEnv(spirv::WorldRayDirectionKHR);
    payload.albedo = _sample.albedo;
    payload.worldNormal = _sample.normal;
}