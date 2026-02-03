#include "common.hlsl"

#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

[shader("anyhit")]
void main(inout PrimaryPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = spirv::InstanceCustomIndexKHR;
    const static uint64_t STriangleGeomInfoAlignment = nbl::hlsl::alignment_of_v<STriangleGeomInfo>;
    const STriangleGeomInfo geom = vk::BufferPointer<STriangleGeomInfo, STriangleGeomInfoAlignment>(pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo)).Get();

    const uint32_t bitpattern = payload.pcg();
    // Cannot use spirv::ignoreIntersectionKHR and spirv::terminateRayKHR due to https://github.com/microsoft/DirectXShaderCompiler/issues/7279
    if (geom.material.alphaTest(bitpattern))
        IgnoreHit();
}
