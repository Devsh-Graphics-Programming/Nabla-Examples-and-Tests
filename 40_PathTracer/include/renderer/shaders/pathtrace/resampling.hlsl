#ifndef _NBL_THIS_EXAMPLE_PATHTRACE_RESAMPLING_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATHTRACE_RESAMPLING_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

namespace nbl
{
namespace this_example
{

struct PathFlags
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Active = 0x0001;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t Specular = 0x0002;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Delta = 0x0004;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiffuseBounce = 0x0008;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SpecularBounce = 0x0010;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t LastVertexlightSampled = 0x0020;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t PathHasSpecularBounce = 0x0040;
};

struct SPathState
{
    uint16_t flags;
    uint32_t currentVertexIndex;    // vertexIndex along the path, intialized value is 1
    uint32_t rcVertexLength;

    bool isLastVertexRough;

    // probably don't need
    // float32_t3 origin;
    float32_t3 normal;
    float32_t3 direction;
    
    float32_t pdf;

    // TODO ReSTIR: double check usage, we might already do all this in ray
    float32_t3 prefixThroughput;
    float32_t3 throughput;
    float32_t3 radiance;

    // float32_t3 LoForDelta;   // TODO ReSTIR: maybe do later, with delta distributions?

    float32_t3 prefixPathRadiance;
    float32_t3 rcVertexRadiance;

    // TODO ReSTIR: pack this? it's the same as SClosestHitInfo
    float32_t3 preRcHitPosition;
    float32_t2 preRcVertexBarycentrics;
    uint32_t preRcVertexInstancedGeometryID;
    uint32_t preRcVertexPrimitiveID;
    float32_t3 preRcNormal;
    float32_t3 preRcVertexL;

    float32_t3 rcVertexPosition;
    float32_t3 rcVertexNormal;
    float32_t rcPdf;

    uint64_t2 rngSeed;

    static SPathState create(uint32_t2 pixel, uint16_t depth)
    {
        SPathState retval;
        retval.currentVertexIndex = 0;
        retval.rcVertexLength = depth;

        retval.isLastVertexRough = false;

        retval.prefixThroughput = 1.f;
        retval.throughput = 1.f;
        retval.flags = PathFlags::Active;

        // TODO: get seed from scramblebuf based on pixel
        // retval.rngSeed = 0;
        return retval;
    }

    void updatePrefixThroughput()
    {
        prefixThroughput *= throughput;
        throughput = 1.f;
    }
};

struct SReconnectionData
{
    float32_t3 preRcHitPosition;
    float32_t2 preRcVertexBarycentrics;
    uint32_t preRcVertexInstancedGeometryID;
    uint32_t preRcVertexPrimitiveID;
    float32_t3 preRcNormal;
    float3_t pathPreRcThroughput;            // indicates path throughput before the preRcVertex
    float3_t pathPreRcRadiance;
    float3_t preRcVertexL;
    uint32_t pathLength;
};

struct SHashAppendData
{
    uint32_t isValid;
    uint32_t reservoirIdx;
    uint32_t cellIdx;
    uint32_t inCellIdx;
};

// TODO ReSTIR: the paper claims that reservoir data is packed accordingly -- maybe try at some point?
// radiance: half-precision float
// position + normal: compressed into single float4
// sample count + age: compressed in 4 byte uint
struct SReservoir
{
    float32_t3 vPosition;
    float32_t3 vNormal;
    float32_t3 sPosition;
    float32_t3 sNormal;
    float32_t3 radiance;

    uint16_t M;
    float32_t weightF;  // used for final illuminance computation W = weight / (M * pdf)
    uint16_t age;   // sample age, discard if > maxSampleAge

    static SReservoir create(NBL_CONST_REF_ARG(SPathState) state)
    {
        SReservoir retval;

        retval.vPosition = state.preRcHitPosition;
        retval.vNormal = state.preRcNormal;
        retval.sPosition = state.rcVertexPosition;
        retval.sNormal = state.rcVertexNormal;
        retval.radiance = pathState.rcVertexRadiance;

        retval.weightF = hlsl::mix(float32_t(0.0), float32_t(1.0) / state.pdf, state.pdf > float32_t(0.0));
        retval.M = uint16_t(1u);
        retval.age = uint16_t(0u);

        return retval;
    }

    bool merge(NBL_CONST_REF_ARG(SReservoir) other, float32_t rand, float32_t pdf, NBL_REF_ARG(float32_t) weightS)
    {
        float32_t weight = other.M * hlsl::max(float32_t(0.0), other.weightF) * pdf;

        weightS += weight;
        M += other.M;
    
        bool isUpdate = rand * weightS <= weight;
        if (isUpdate)
        {
            sPosition = other.sPosition;
            sNormal = other.sNormal;
            radiance = other.radiance;
            age = other.age;
        }
        return isUpdate;
    }

    void updateFinalWeight(float32_t targetPdf, float32_t weightS)
    {
        float32_t weight = targetPdf * M;
        weightF = hlsl::mix(float32_t(0.0), weightS / weight, weight > float32_t(0.0));
    }
};

}
}


#endif  // _NBL_THIS_EXAMPLE_PATHTRACE_RESAMPLING_HLSL_INCLUDED_
