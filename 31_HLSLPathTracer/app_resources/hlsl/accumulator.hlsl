#ifndef _NBL_HLSL_EXT_ACCUMULATOR_INCLUDED_
#define _NBL_HLSL_EXT_ACCUMULATOR_INCLUDED_

#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>

namespace Accumulator
{

template<typename OutputTypeVec NBL_PRIMARY_REQUIRES(concepts::FloatingPointVector<OutputTypeVec>)
struct DefaultAccumulator
{
    using output_storage_type = OutputTypeVec;
    using this_t = DefaultAccumulator<OutputTypeVec>;
    output_storage_type accumulation;

    static this_t create()
    {
        this_t retval;
        retval.accumulation = promote<OutputTypeVec, float32_t>(0.0f);

        return retval;
    }

    void addSample(uint32_t sampleCount, float32_t3 sample)
    {
        using ScalarType = typename vector_traits<OutputTypeVec>::scalar_type;
        ScalarType rcpSampleSize = 1.0 / (sampleCount);
        accumulation += (sample - accumulation) * rcpSampleSize;
    }
};

}

#endif
