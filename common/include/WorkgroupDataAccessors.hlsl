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

template<class Config, class Binop>
struct DataProxy
{
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;

    static DataProxy<Config, Binop> create()
    {
        DataProxy<Config, Binop> retval;
        retval.workgroupOffset = glsl::gl_WorkGroupID().x * Config::VirtualWorkgroupSize;
        retval.outputBufAddr = sizeof(uint32_t) + vk::RawBufferLoad<uint64_t>(pc.ppOutputBuf + Binop::BindingIndex * sizeof(uint64_t));
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = vk::RawBufferLoad<AccessType>(pc.pInputBuf + (workgroupOffset + ix) * sizeof(AccessType));
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        vk::RawBufferStore<AccessType>(outputBufAddr + sizeof(AccessType) * (workgroupOffset+ix), value, sizeof(uint32_t));
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    uint32_t workgroupOffset;
    uint64_t outputBufAddr;
};

template<class Config, class Binop>
struct PreloadedDataProxy
{
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t PreloadedDataCount = Config::VirtualWorkgroupSize / Config::WorkgroupSize;

    static PreloadedDataProxy<Config, Binop> create()
    {
        PreloadedDataProxy<Config, Binop> retval;
        retval.data = DataProxy<Config, Binop>::create();
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = preloaded[ix>>Config::WorkgroupSizeLog2];
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        preloaded[ix>>Config::WorkgroupSizeLog2] = value;
    }

    void preload()
    {
        const uint16_t invocationIndex = workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template get<dtype_t, uint16_t>(idx * Config::WorkgroupSize + invocationIndex, preloaded[idx]);
    }
    void unload()
    {
        const uint16_t invocationIndex = workgroup::SubgroupContiguousIndex();
        [unroll]
        for (uint16_t idx = 0; idx < PreloadedDataCount; idx++)
            data.template set<dtype_t, uint16_t>(idx * Config::WorkgroupSize + invocationIndex, preloaded[idx]);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DataProxy<Config, Binop> data;
    dtype_t preloaded[PreloadedDataCount];
};

}
}

#endif
