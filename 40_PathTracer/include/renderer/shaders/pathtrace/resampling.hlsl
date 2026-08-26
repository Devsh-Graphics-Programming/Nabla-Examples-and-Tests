#ifndef _NBL_THIS_EXAMPLE_PATHTRACE_RESAMPLING_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATHTRACE_RESAMPLING_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

namespace nbl
{
namespace this_example
{

NBL_CONSTEXPR uint32_t HashGridWorkgroupSize = 16u;

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
    hlsl::float32_t3 normal;
    hlsl::float32_t3 direction;
    
    hlsl::float32_t pdf;

    // TODO ReSTIR: double check usage, we might already do all this in ray
    hlsl::float32_t3 prefixThroughput;
    hlsl::float32_t3 throughput;
    hlsl::float32_t3 radiance;

    // float32_t3 LoForDelta;   // TODO ReSTIR: maybe do later, with delta distributions?

    hlsl::float32_t3 prefixPathRadiance;
    hlsl::float32_t3 rcVertexRadiance;

    // TODO ReSTIR: pack this? it's the same as SClosestHitInfo
    hlsl::float32_t3 preRcHitPosition;
    hlsl::float32_t2 preRcVertexBarycentrics;
    uint32_t preRcVertexInstancedGeometryID;
    uint32_t preRcVertexPrimitiveID;
    hlsl::float32_t3 preRcNormal;
    hlsl::float32_t3 preRcVertexL;

    hlsl::float32_t3 rcVertexPosition;
    hlsl::float32_t3 rcVertexNormal;
    hlsl::float32_t rcPdf;

    hlsl::uint64_t2 rngSeed;

    static SPathState create(hlsl::uint32_t2 pixel, uint16_t depth)
    {
        SPathState retval;
        retval.currentVertexIndex = 0;
        retval.rcVertexLength = uint32_t(depth);

        retval.isLastVertexRough = false;

        retval.prefixThroughput = hlsl::promote<hlsl::float32_t3>(1.f);
        retval.throughput = hlsl::promote<hlsl::float32_t3>(1.f);
        retval.flags = PathFlags::Active;

        // TODO: get seed from scramblebuf based on pixel
        // retval.rngSeed = 0;
        return retval;
    }

    void updatePrefixThroughput()
    {
        prefixThroughput *= throughput;
        throughput = hlsl::promote<hlsl::float32_t3>(1.f);
    }
};

struct SReconnectionData
{
    hlsl::float32_t3 preRcHitPosition;
    hlsl::float32_t2 preRcVertexBarycentrics;
    uint32_t preRcVertexInstancedGeometryID;
    uint32_t preRcVertexPrimitiveID;
    hlsl::float32_t3 preRcNormal;
    hlsl::float32_t pathPreRcThroughput;            // indicates path throughput before the preRcVertex
    hlsl::float32_t pathPreRcRadiance;
    hlsl::float32_t preRcVertexL;
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
    hlsl::float32_t3 vPosition;
    hlsl::float32_t3 vNormal;
    hlsl::float32_t3 sPosition;
    hlsl::float32_t3 sNormal;
    hlsl::float32_t3 radiance;

    uint16_t M;
    uint16_t age;   // sample age, discard if > maxSampleAge
    hlsl::float32_t weightF;  // used for final illuminance computation W = weight / (M * pdf)

    static SReservoir create(NBL_CONST_REF_ARG(SPathState) state)
    {
        SReservoir retval;

        retval.vPosition = state.preRcHitPosition;
        retval.vNormal = state.preRcNormal;
        retval.sPosition = state.rcVertexPosition;
        retval.sNormal = state.rcVertexNormal;
        retval.radiance = state.rcVertexRadiance;

        // retval.weightF = hlsl::mix(hlsl::float32_t(0.0), hlsl::float32_t(1.0) / state.pdf, state.pdf > hlsl::float32_t(0.0));
        retval.weightF = hlsl::mix(0.0f, 0.1f, state.pdf > hlsl::float32_t(0.0));
        retval.M = uint16_t(1u);
        retval.age = uint16_t(0u);

        return retval;
    }

    bool merge(NBL_CONST_REF_ARG(SReservoir) other, hlsl::float32_t rand, hlsl::float32_t pdf, NBL_REF_ARG(hlsl::float32_t) weightS)
    {
        hlsl::float32_t weight = other.M * hlsl::max(0.0f, other.weightF) * pdf;

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

    void updateFinalWeight(hlsl::float32_t targetPdf, hlsl::float32_t weightS)
    {
        hlsl::float32_t weight = targetPdf * M;
        weightF = hlsl::mix(hlsl::float32_t(0.0), weightS / weight, weight > hlsl::float32_t(0.0));
    }
};

}
}

#endif  // _NBL_THIS_EXAMPLE_PATHTRACE_RESAMPLING_HLSL_INCLUDED_
