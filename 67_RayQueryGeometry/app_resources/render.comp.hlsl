#include "common.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"


using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;

[[vk::binding(1, 0)]] RWTexture2D<float4> outImage;
[[vk::constant_id(0)]] const float shader_variant = 1.0;

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
    const uint indexType = geom.indexType;
    const uint vertexStride = geom.vertexStride;
    const uint objType = instID;

    const uint64_t vertexBufferAddress = geom.vertexBufferAddress;
    const uint64_t indexBufferAddress = geom.indexBufferAddress;
    const uint64_t normalBufferAddress = geom.normalBufferAddress;

    uint32_t3 indices;
    switch (indexType)
    {
        case 0: // EIT_16BIT
            indices = uint32_t3((nbl::hlsl::bda::__ptr<uint16_t3>::create(indexBufferAddress)+primID).deref().load());
            break;
        case 1: // EIT_32BIT
            indices = uint32_t3((nbl::hlsl::bda::__ptr<uint32_t3>::create(indexBufferAddress)+primID).deref().load());
            break;
        default:    // EIT_NONE
        {
            indices[0] = primID * 3;
            indices[1] = indices[0] + 1;
            indices[2] = indices[0] + 2;
        }
    }

    float3 n0, n1, n2;
    switch (objType)
    {
        case OT_CUBE:
        case OT_SPHERE:
        case OT_RECTANGLE:
        case OT_CYLINDER:
        //case OT_ARROW:
        case OT_CONE:
        {
            // TODO: document why the alignment is 2 here and nowhere else? isnt the `vertexStride` aligned to more than 2 anyway?
            uint32_t v0 = vk::RawBufferLoad<uint32_t>(normalBufferAddress + indices[0] * 4);
            uint32_t v1 = vk::RawBufferLoad<uint32_t>(normalBufferAddress + indices[1] * 4);
            uint32_t v2 = vk::RawBufferLoad<uint32_t>(normalBufferAddress + indices[2] * 4);

            n0 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v0).xyz);
            n1 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v1).xyz);
            n2 = normalize(nbl::hlsl::spirv::unpackSnorm4x8(v2).xyz);
        }
        break;
        case OT_ICOSPHERE:
        default:
        {
            n0 = normalize(vk::RawBufferLoad<float3>(normalBufferAddress + indices[0] * vertexStride));
            n1 = normalize(vk::RawBufferLoad<float3>(normalBufferAddress + indices[1] * vertexStride));
            n2 = normalize(vk::RawBufferLoad<float3>(normalBufferAddress + indices[2] * vertexStride));
        }
    }

    float3 barycentrics = float3(0.0, bary);
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;        

    return barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;
}

[numthreads(WorkgroupSize, WorkgroupSize, 1)]
[shader("compute")]
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

        // TODO: candidate for `bda::__ptr<SGeomInfo>`
        const SGeomInfo geom = vk::RawBufferLoad<SGeomInfo>(pc.geometryInfoBuffer + instID * sizeof(SGeomInfo),8);

        float3 normals;
        float2 barycentrics = spirv::rayQueryGetIntersectionBarycentricsKHR(query, true);
        normals = calculateSmoothNormals(instID, primID, geom, barycentrics);

        normals = normalize(normals) * 0.5 + 0.5;
        color = float4(normals, 1.0);
    }

    outImage[threadID.xy] = color;
}
