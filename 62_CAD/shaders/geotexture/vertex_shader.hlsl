#pragma shader_stage(vertex)

#include "common.hlsl"

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    
    PSInput outV;
    ClipProjectionData clipProjectionData = globals.defaultClipProjection;
    outV.position.z = 0.0;

    const float32_t2 dirV = float32_t2(geoTextureOBB.dirU.y, -geoTextureOBB.dirU.x) * geoTextureOBB.aspectRatio;
    const float2 screenTopLeft = _static_cast<float2>(transformPointNdc(clipProjectionData.projectionToNDC, geoTextureOBB.topLeft));
    const float2 screenDirU = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, _static_cast<pfloat64_t2>(geoTextureOBB.dirU)));
    const float2 screenDirV = _static_cast<float2>(transformVectorNdc(clipProjectionData.projectionToNDC, _static_cast<pfloat64_t2>(dirV)));

    const float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1)); // corners of square from (0, 0) to (1, 1)

    const float2 coord = screenTopLeft + corner.x * screenDirU + corner.y * screenDirV;
    outV.position.xy = coord;
    outV.uv = corner;

    return outV;
}
