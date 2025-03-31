#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("anyhit")]
void main(inout PrimaryPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const int instID = InstanceID();
    const STriangleGeomInfo geom = vk::RawBufferLoad < STriangleGeomInfo > (pc.triangleGeomInfoBuffer + instID * sizeof(STriangleGeomInfo));

    // Should have been a method of the payload but https://github.com/microsoft/DirectXShaderCompiler/issues/6464 stops it
    // alpha is quantized to 10 bits
    const uint32_t bitpattern = payload.pcg()>>22;
    if (bitpattern > geom.material.alpha)
        IgnoreHit();
}
