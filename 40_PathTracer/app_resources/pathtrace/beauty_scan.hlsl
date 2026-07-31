#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"
#include "nbl/builtin/hlsl/scan/chained_scan.hlsl"

#include "common.hlsl"
#include "renderer/shaders/bda_accessors.hlsl"

using config_t = WORKGROUP_CONFIG_T;

[[vk::push_constant]] SScanPushConstants pc;

struct device_capabilities
{
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = true;
};

typedef vector<uint32_t, config_t::ItemsPerInvocation_0> type_t;

groupshared uint32_t scratch[mpl::max_v<int16_t,config_t::SharedScratchElementCount,1>];

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
        retval.inputAddress = inputBuf;
        retval.outputAddress = outputBuf;
        return retval;
    }

    void initAtWorkgroupID(const uint32_t workgroupID)
    {
        const uint32_t workgroupOffset = workgroupID * VirtualWorkgroupSize * sizeof(dtype_t);
        accessor = DoubleLegacyBdaAccessor<dtype_t>::create(inputAddress + workgroupOffset, outputAddress + workgroupOffset);
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

    uint64_t getInputBufAddr()
    {
        return inputAddress;
    }
    uint64_t getOutputBufAddr()
    {
        return outputAddress;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
        //glsl::memoryBarrierShared(); implied by the above
    }

    DoubleLegacyBdaAccessor<dtype_t> accessor;
    uint64_t inputAddress, outputAddress;
};

template<typename T>
struct ReduceAccessor
{
    using type_t = T;
    static ReduceAccessor<T> create(const uint64_t addr)
    {
        ReduceAccessor<T> retval;
        retval.ptr = bda::__ptr<uint32_t>::create(addr);
        return retval;
    }

    template<typename AccessType, typename IndexType>
    void get(const IndexType index, NBL_REF_ARG(AccessType) value)
    {
        bda::__ptr<T> target = ptr + index;
        value = target.template deref().load();
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType index, const AccessType value)
    {
        bda::__ptr<T> target = ptr + index;
        return target.template deref().store(value);
    }

    T atomicMax(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = ptr + index;
        return glsl::atomicMax(target.template deref().ptr.value, value);
    }
    T atomicExchange(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = ptr + index;
        return glsl::atomicExchange(target.template deref().ptr.value, value);
    }

    bda::__ptr<T> ptr;
};

struct WorkgroupCounter
{
    static WorkgroupCounter create(const uint64_t addr)
    {
        WorkgroupCounter retval;
        retval.ptr = bda::__ptr<uint32_t>::create(addr);
        return retval;
    }

    uint32_t atomicAdd(const uint64_t index, const uint32_t value)
    {
        bda::__ptr<uint32_t> target = ptr + index;
        return glsl::atomicAdd(target.template deref().ptr.value, value);
    }

    bda::__ptr<uint32_t> ptr;
};

static ScratchProxy arithmeticAccessor;

[numthreads(config_t::WorkgroupSize, config_t::WorkgroupSize, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	using data_proxy_t = DataProxy<config_t::VirtualWorkgroupSize,config_t::ItemsPerInvocation_0>;
    data_proxy_t dataAccessor = data_proxy_t::create(pc.pInputBuf, pc.pOutputBuf);

    using reduce_proxy_t = ReduceAccessor<uint32_t>;
    reduce_proxy_t reduceAccessor = reduce_proxy_t::create(pc.pReduceBuf);

    WorkgroupCounter wgCounter = WorkgroupCounter::create(pc.pWgCounterBuf);

    // TODO: double check but I think it's inclusive scan, not exclusive
    using binop_t = nbl::hlsl::plus<uint32_t>;
    nbl::hlsl::scan::inclusive_scan<config_t,binop_t,device_capabilities>::template __call<data_proxy_t, ScratchProxy, reduce_proxy_t, WorkgroupCounter>(dataAccessor,arithmeticAccessor,reduceAccessor,wgCounter);
    // we barrier before because we alias the accessors for Binop
    arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
}