#ifndef _NBL_HLSL_EXT_ACCUMULATOR_INCLUDED_
#define _NBL_HLSL_EXT_ACCUMULATOR_INCLUDED_

#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>

namespace Accumulator
{

template<typename OutputTypeVec NBL_PRIMARY_REQUIRES(concepts::FloatingPointVector<OutputTypeVec>)
struct DefaultAccumulator
{
    using input_sample_type = OutputTypeVec;
    using output_storage_type = OutputTypeVec;
    using this_t = DefaultAccumulator<OutputTypeVec>;
    using scalar_type = typename vector_traits<OutputTypeVec>::scalar_type;

    static this_t create()
    {
        this_t retval;
        retval.accumulation = promote<OutputTypeVec, scalar_type>(0.0f);

        return retval;
    }

    void addSample(uint32_t sampleCount, input_sample_type _sample)
    {
        scalar_type rcpSampleSize = 1.0 / (sampleCount);
        accumulation += (_sample - accumulation) * rcpSampleSize;
    }

    output_storage_type accumulation;
};

}

#endif
