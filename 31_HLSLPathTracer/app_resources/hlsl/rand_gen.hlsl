#ifndef _PATHTRACER_EXAMPLE_RANDGEN_INCLUDED_
#define _PATHTRACER_EXAMPLE_RANDGEN_INCLUDED_

#include "nbl/builtin/hlsl/sampling/quantized_sequence.hlsl"
#include "nbl/builtin/hlsl/random/dim_adaptor_recursive.hlsl"

#include "render_common.hlsl"

using namespace nbl;
using namespace hlsl;

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
        using sequence_type = sampling::QuantizedSequence<uint32_t2,3>;
        uint32_t address = glsl::bitfieldInsert<uint32_t>(baseDimension, sampleIndex, MaxDepthLog2, MaxSamplesLog2);
        sequence_type tmpSeq = vk::RawBufferLoad<sequence_type>(pSampleBuffer + address * sizeof(sequence_type));
        return tmpSeq.template decode<float32_t>(random::DimAdaptorRecursive<rng_type, N>::__call(rng));
    }

    rng_type rng;
    uint64_t pSampleBuffer;
};

#endif
