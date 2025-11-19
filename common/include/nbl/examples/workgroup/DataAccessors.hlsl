#ifndef _NBL_EXAMPLES_WORKGROUP_DATA_ACCESSORS_HLSL_
#define _NBL_EXAMPLES_WORKGROUP_DATA_ACCESSORS_HLSL_


#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"


namespace nbl
{
namespace hlsl
{
namespace examples
{
namespace workgroup
{

struct ScratchProxy
{
    template<typename AccessType, typename IndexType>
    void get(const uint32_t ix, NBL_REF_ARG(AccessType) value)
    {
        value = scratch[ix];
    }
    template<typename AccessType, typename IndexType>
    void set(const uint32_t ix, const AccessType value)
    {
        scratch[ix] = value;
    }

    uint32_t atomicOr(const uint32_t ix, const uint32_t value)
    {
        return glsl::atomicOr(scratch[ix],value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }
};

template<uint16_t VirtualWorkgroupSize, uint16_t ItemsPerInvocation>
struct DataProxy
{
    using dtype_t = vector<uint32_t, ItemsPerInvocation>;
    // function template AccessType should be the same as dtype_t

    static DataProxy<VirtualWorkgroupSize, ItemsPerInvocation> create(const uint64_t inputBuf, const uint64_t outputBuf)
    {
        DataProxy<VirtualWorkgroupSize, ItemsPerInvocation> retval;
        const uint32_t workgroupOffset = glsl::gl_WorkGroupID().x * VirtualWorkgroupSize * sizeof(dtype_t);
        retval.accessor = DoubleLegacyBdaAccessor<dtype_t>::create(inputBuf + workgroupOffset, outputBuf + workgroupOffset);
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        accessor.get(ix, value);
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        accessor.set(ix, value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DoubleLegacyBdaAccessor<dtype_t> accessor;
};

template<uint16_t WorkgroupSizeLog2, uint16_t VirtualWorkgroupSize, uint16_t ItemsPerInvocation>
struct PreloadedDataProxy
{
    using dtype_t = vector<uint32_t, ItemsPerInvocation>;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1u) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t PreloadedDataCount = VirtualWorkgroupSize / WorkgroupSize;

    static PreloadedDataProxy<WorkgroupSizeLog2, VirtualWorkgroupSize, ItemsPerInvocation> create(const uint64_t inputBuf, const uint64_t outputBuf)
    {
        PreloadedDataProxy<WorkgroupSizeLog2, VirtualWorkgroupSize, ItemsPerInvocation> retval;
        retval.data = DataProxy<VirtualWorkgroupSize, ItemsPerInvocation>::create(inputBuf, outputBuf);
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = preloaded[ix>>WorkgroupSizeLog2];
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        preloaded[ix>>WorkgroupSizeLog2] = value;
    }

    void preload()
    {
        const uint16_t invocationIndex = hlsl::workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template get<dtype_t, uint16_t>(idx * WorkgroupSize + invocationIndex, preloaded[idx]);
    }
    void unload()
    {
        const uint16_t invocationIndex = hlsl::workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template set<dtype_t, uint16_t>(idx * WorkgroupSize + invocationIndex, preloaded[idx]);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DataProxy<VirtualWorkgroupSize, ItemsPerInvocation> data;
    dtype_t preloaded[PreloadedDataCount];
};

}
}
}
}
#endif
