#include "common.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

[[vk::binding(1, 0)]] RWTexture2D<float4> outImage;

float3 unpackNormals3x10(uint32_t v)
{
    // host side changes float32_t3 to EF_A2B10G10R10_SNORM_PACK32
    // follows unpacking scheme from https://github.com/KhronosGroup/SPIRV-Cross/blob/main/reference/shaders-hlsl/frag/unorm-snorm-packing.frag
    int signedValue = int(v);
    int3 pn = int3(signedValue << 22, signedValue << 12, signedValue << 2) >> 22;
    return clamp(float3(pn) / 511.0, -1.0, 1.0);
}

float3 calculateSmoothNormals(int instID, int primID, SGeomInfo geom, float2 bary)
{
    uint idxOffset = primID * 3;

    const uint indexType = geom.indexType;
    const uint vertexStride = geom.vertexStride;

    const uint64_t vertexBufferAddress = geom.vertexBufferAddress;
    const uint64_t indexBufferAddress = geom.indexBufferAddress;

    uint i0, i1, i2;
    switch (indexType)
    {
        case 0: // EIT_16BIT
        {
            i0 = uint32_t(vk::RawBufferLoad<uint16_t>(indexBufferAddress + (idxOffset + 0) * sizeof(uint16_t), 2u));
            i1 = uint32_t(vk::RawBufferLoad<uint16_t>(indexBufferAddress + (idxOffset + 1) * sizeof(uint16_t), 2u));
            i2 = uint32_t(vk::RawBufferLoad<uint16_t>(indexBufferAddress + (idxOffset + 2) * sizeof(uint16_t), 2u));
        }
        break;
        case 1: // EIT_32BIT
        {
            i0 = vk::RawBufferLoad<uint32_t>(indexBufferAddress + (idxOffset + 0) * sizeof(uint32_t));
            i1 = vk::RawBufferLoad<uint32_t>(indexBufferAddress + (idxOffset + 1) * sizeof(uint32_t));
            i2 = vk::RawBufferLoad<uint32_t>(indexBufferAddress + (idxOffset + 2) * sizeof(uint32_t));
        }
        break;
        default:    // EIT_NONE
        {
            i0 = idxOffset;
            i1 = idxOffset + 1;
            i2 = idxOffset + 2;
        }
    }

    float3 n0, n1, n2;
    switch (instID)
    {
        case OT_CUBE:
        {
            uint32_t v0 = vk::RawBufferLoad<uint32_t>(vertexBufferAddress + i0 * vertexStride, 2u);
            uint32_t v1 = vk::RawBufferLoad<uint32_t>(vertexBufferAddress + i1 * vertexStride, 2u);
            uint32_t v2 = vk::RawBufferLoad<uint32_t>(vertexBufferAddress + i2 * vertexStride, 2u);

            n0 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v0).xyz);
            n1 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v1).xyz);
            n2 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v2).xyz);
        }
        break;
        case OT_SPHERE:
        case OT_CYLINDER:
        case OT_ARROW:
        case OT_CONE:
        {
            uint32_t v0 = vk::RawBufferLoad<uint32_t>(vertexBufferAddress + i0 * vertexStride);
            uint32_t v1 = vk::RawBufferLoad<uint32_t>(vertexBufferAddress + i1 * vertexStride);
            uint32_t v2 = vk::RawBufferLoad<uint32_t>(vertexBufferAddress + i2 * vertexStride);

            n0 = normalize(unpackNormals3x10(v0));
            n1 = normalize(unpackNormals3x10(v1));
            n2 = normalize(unpackNormals3x10(v2));
        }
        break;
        case OT_RECTANGLE:
        case OT_DISK:
        case OT_ICOSPHERE:
        default:
        {
            n0 = normalize(vk::RawBufferLoad<float3>(vertexBufferAddress + i0 * vertexStride));
            n1 = normalize(vk::RawBufferLoad<float3>(vertexBufferAddress + i1 * vertexStride));
            n2 = normalize(vk::RawBufferLoad<float3>(vertexBufferAddress + i2 * vertexStride));
        }
    }

    float3 barycentrics = float3(0.0, bary);
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;        

    return barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;
}

[numthreads(WorkgroupSize, WorkgroupSize, 1)]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    uint2 coords = threadID.xy;
    coords.y = nbl::hlsl::glsl::gl_NumWorkGroups().y * WorkgroupSize - coords.y;    // need to invert it
    
    float4 NDC;
    NDC.xy = float2(coords) * pc.scaleNDC;
    NDC.xy += pc.offsetNDC;
    NDC.zw = float2(0, 1);
    float3 targetPos;
    {
        float4 tmp = mul(pc.invMVP, NDC);
        targetPos = tmp.xyz / tmp.w;
    }

    float3 direction = normalize(targetPos - pc.camPos);

    spirv::RayQueryKHR query;
    spirv::rayQueryInitializeKHR(query, topLevelAS, spv::RayFlagsOpaqueKHRMask, 0xFF, pc.camPos, 0.01, direction, 1000.0);

    while (spirv::rayQueryProceedKHR(query)) {}

    float4 color = float4(0, 0, 0, 1);

    if (spirv::rayQueryGetIntersectionTypeKHR(query, true) == spv::RayQueryCommittedIntersectionTypeRayQueryCommittedIntersectionTriangleKHR)
    {
        const int instID = spirv::rayQueryGetIntersectionInstanceIdKHR(query, true);
        const int primID = spirv::rayQueryGetIntersectionPrimitiveIndexKHR(query, true);

        const SGeomInfo geom = vk::RawBufferLoad<SGeomInfo>(pc.geometryInfoBuffer + instID * sizeof(SGeomInfo));
        
        float3 normals;
        if (jit::device_capabilities::rayTracingPositionFetch)
        {
            if (geom.smoothNormals)
            {
                float2 barycentrics = spirv::rayQueryGetIntersectionBarycentricsKHR(query, true);
                normals = calculateSmoothNormals(instID, primID, geom, barycentrics);
            }
            else
            {
                float3 pos[3] = spirv::rayQueryGetIntersectionTriangleVertexPositionsKHR(query, true);
                normals = cross(pos[1] - pos[0], pos[2] - pos[0]);
            }
        }
        else
        {
            float2 barycentrics = spirv::rayQueryGetIntersectionBarycentricsKHR(query, true);
            normals = calculateSmoothNormals(instID, primID, geom, barycentrics);
        }

        normals = normalize(normals) * 0.5 + 0.5;
        color = float4(normals, 1.0);
    }

    outImage[threadID.xy] = color;
}
