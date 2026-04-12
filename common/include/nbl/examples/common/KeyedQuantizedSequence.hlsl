#ifndef _NBL_EXAMPLES_KEYED_QUANTIZED_SEQUENCE_HLSL_
#define _NBL_EXAMPLES_KEYED_QUANTIZED_SEQUENCE_HLSL_


#include "nbl/builtin/hlsl/sampling/quantized_sequence.hlsl"
#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"


namespace nbl
{
namespace hlsl
{
namespace examples
{

template<typename RNG=Xoroshiro64Star>
struct KeyedQuantizedSequence
{
    using rng_type = RNG; // legacy
    using key_rng_type = RNG;
    using sequence_type = hlsl::sampling::QuantizedSequence<uint32_t2,3>;
    using return_type = vector<float32_t,3>;

    // baseDimension: offset index of the sequence
    // sampleIndex: iteration number of current pixel (samples per pixel)
    return_type operator()(uint32_t baseDimension, const uint32_t sampleIndex)
    {
        const uint32_t address = sampleIndex|(baseDimension<<sequenceSamplesLog2);
        sequence_type tmpSeq = vk::RawBufferLoad<sequence_type>(pSampleBuffer + address * sizeof(sequence_type));
        sequence_type scramble;
        scramble.data[0] = rng();
        scramble.data[1] = rng();
        return tmpSeq.template decode<float32_t>(scramble);
    }

    // could be vk::BufferPointer<sequence_type> but no arithmetic
    uint64_t pSampleBuffer;
    key_rng_type rng;
    uint16_t sequenceSamplesLog2;
};

}
}
}
#endif
