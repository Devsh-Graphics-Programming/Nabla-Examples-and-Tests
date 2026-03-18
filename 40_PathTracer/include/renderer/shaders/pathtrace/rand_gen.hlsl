#ifndef _NBL_THIS_EXAMPLE_PATHTRACE_RANDGEN_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATHTRACE_RANDGEN_HLSL_INCLUDED_

#include "renderer/shaders/pathtrace/common.hlsl"

#include "nbl/builtin/hlsl/sampling/quantized_sequence.hlsl"
#include "nbl/builtin/hlsl/random/dim_adaptor_recursive.hlsl"

namespace nbl
{
namespace this_example
{

template<typename RNG, uint16_t N>
struct RandomUniformND
{
    using rng_type = RNG;
    using return_type = vector<float32_t, N>;

    static RandomUniformND<RNG,N> create(uint32_t2 seed, uint64_t pSampleSequence)
    {
        RandomUniformND<RNG,N> retval;
        retval.rng = rng_type::construct(seed);
        retval.pSampleBuffer = pSampleSequence;
        return retval;
    }

    // baseDimension: offset index of the sequence
    // sampleIndex: iteration number of current pixel (samples per pixel)
    return_type operator()(uint32_t baseDimension, uint32_t sampleIndex)
    {
        using sequence_type = hlsl::sampling::QuantizedSequence<uint32_t2,3>;
        uint32_t address = hlsl::glsl::bitfieldInsert<uint32_t>(baseDimension, sampleIndex, SSensorUniforms::MaxPathDepthLog2, SSensorUniforms::MaxSamplesLog2);
        sequence_type tmpSeq = vk::RawBufferLoad<sequence_type>(pSampleBuffer + address * sizeof(sequence_type));
        return tmpSeq.template decode<float32_t>(hlsl::random::DimAdaptorRecursive<rng_type, N>::__call(rng));
    }

    rng_type rng;
    uint64_t pSampleBuffer;
};

}
}

#endif
