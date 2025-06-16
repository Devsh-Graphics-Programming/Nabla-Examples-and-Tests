#ifndef _WORKGROUP_DATA_ACCESSORS_HLSL_
#define _WORKGROUP_DATA_ACCESSORS_HLSL_

namespace nbl
{
namespace hlsl
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

template<uint16_t WorkgroupSize, uint16_t ItemsPerInvocation>
struct DataProxy
{
    using dtype_t = vector<uint32_t, ItemsPerInvocation>;

    static DataProxy<WorkgroupSize, ItemsPerInvocation> create(uint64_t inputBuf, uint64_t outputBuf)
    {
        DataProxy<WorkgroupSize, ItemsPerInvocation> retval;
        retval.workgroupOffset = glsl::gl_WorkGroupID().x * WorkgroupSize;
        retval.inputBufAddr = inputBuf;
        retval.outputBufAddr = outputBuf;
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = vk::RawBufferLoad<AccessType>(inputBufAddr + (workgroupOffset + ix) * sizeof(AccessType));
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        vk::RawBufferStore<AccessType>(outputBufAddr + (workgroupOffset + ix) * sizeof(AccessType), value, sizeof(uint32_t));
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    uint32_t workgroupOffset;
    uint64_t inputBufAddr;
    uint64_t outputBufAddr;
};

template<uint16_t WorkgroupSizeLog2, uint16_t ItemsPerInvocation, uint16_t _PreloadedDataCount>
struct PreloadedDataProxy
{
    using dtype_t = vector<uint32_t, ItemsPerInvocation>;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t PreloadedDataCount = _PreloadedDataCount;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1u) << WorkgroupSizeLog2;

    static PreloadedDataProxy<WorkgroupSizeLog2, ItemsPerInvocation, PreloadedDataCount> create(uint64_t inputBuf, uint64_t outputBuf)
    {
        PreloadedDataProxy<WorkgroupSizeLog2, ItemsPerInvocation, PreloadedDataCount> retval;
        retval.data = DataProxy<WorkgroupSize*PreloadedDataCount, ItemsPerInvocation>::create(inputBuf, outputBuf);
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
        const uint16_t invocationIndex = workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template get<dtype_t, uint16_t>(idx * WorkgroupSize + invocationIndex, preloaded[idx]);
    }
    void unload()
    {
        const uint16_t invocationIndex = workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template set<dtype_t, uint16_t>(idx * WorkgroupSize + invocationIndex, preloaded[idx]);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DataProxy<WorkgroupSize*PreloadedDataCount, ItemsPerInvocation> data;
    dtype_t preloaded[PreloadedDataCount];
};

}
}

#endif
