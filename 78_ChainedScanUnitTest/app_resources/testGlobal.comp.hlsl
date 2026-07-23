#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"
#include "nbl/builtin/hlsl/scan/chained_scan.hlsl"

using config_t = WORKGROUP_CONFIG_T;

#include "app_resources/shaderCommon.hlsl"

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
    static ReduceAccessor<T> create(const bda::__ptr<T> ptr)
    {
        ReduceAccessor<T> retval;
        retval.ptr = ptr;
        return retval;
    }

    void get(const uint64_t index, NBL_REF_ARG(T) value)
    {
        bda::__ptr<T> target = ptr + index;
        value = target.template deref().load();
    }
    void set(const uint64_t index, const T value)
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

    uint32_t atomicAdd(const uint64_t index, const uint32_t value)  // TODO: maybe it should be just increment
    {
        bda::__ptr<uint32_t> target = ptr + index;
        return glsl::atomicAdd(target.template deref().ptr.value, value);
    }

    bda::__ptr<uint32_t> ptr;
};

static ScratchProxy arithmeticAccessor;

template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    void operator()()
    {
        using data_proxy_t = DataProxy<config_t::VirtualWorkgroupSize,config_t::ItemsPerInvocation_0>;
        data_proxy_t dataAccessor = data_proxy_t::create(pc.pInputBuf, pc.pOutputBuf[Binop::BindingIndex]);

        using reduce_proxy_t = ReduceAccessor<otype_t>;
        bda::__ptr<otype_t> ptr = bda::__ptr<otype_t>::create(pc.pReduceBuf);
        reduce_proxy_t reduceAccessor = reduce_proxy_t::create(ptr);

        WorkgroupCounter wgCounter = WorkgroupCounter::create(pc.pWgCounterBuf);

        OPERATION<config_t,binop_base_t,device_capabilities>::template __call<data_proxy_t, ScratchProxy, reduce_proxy_t, WorkgroupCounter>(dataAccessor,arithmeticAccessor,reduceAccessor,wgCounter);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
    }
};


template<class Binop>
static void subtest()
{
    assert(glsl::gl_SubgroupSize() == config_t::SubgroupSize)

    operation_t<Binop,device_capabilities> func;
    func();
}

void test()
{
    // subtest<arithmetic::bit_and<uint32_t> >();
    // subtest<arithmetic::bit_xor<uint32_t> >();
    // subtest<arithmetic::bit_or<uint32_t> >();
    subtest<arithmetic::plus<uint32_t> >();
    // subtest<arithmetic::multiplies<uint32_t> >();
    // subtest<arithmetic::minimum<uint32_t> >();
    // subtest<arithmetic::maximum<uint32_t> >();
}

[numthreads(config_t::WorkgroupSize,1,1)]
void main()
{
    test();
}
