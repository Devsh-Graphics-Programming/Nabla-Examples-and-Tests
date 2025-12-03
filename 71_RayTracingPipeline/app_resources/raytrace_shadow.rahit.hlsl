#include "common.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

[shader("anyhit")]
void main(inout OcclusionPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = spirv::InstanceCustomIndexKHR;
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));
    const Material material = nbl::hlsl::_static_cast<Material>(geom.material);
    
    const float attenuation = (1.f-material.alpha) * payload.attenuation;
    // DXC cogegens weird things in the presence of termination instructions
    payload.attenuation = attenuation;


    // Cannot use spirv::ignoreIntersectionKHR and spirv::terminateRayKHR due to https://github.com/microsoft/DirectXShaderCompiler/issues/7279
    // arbitrary constant, whatever you want the smallest attenuation to be. Remember until miss, the attenuatio is negative
    if (attenuation > -1.f/1024.f)
        AcceptHitAndEndSearch();
    IgnoreHit();
}
