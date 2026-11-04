#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"
#include "nbl/builtin/hlsl/scan/chained_scan.hlsl"

using config_t = WORKGROUP_CONFIG_T;

#include "app_resources/shaderCommon.hlsl"

// SharedScratchElementCount is 0 when VirtualWorkgroupSize == SubgroupSize (no shared memory needed), so we max with 1 since a 0-sized array won't compile (mpl::max_v is a compile-time max)
groupshared uint32_t scratch[mpl::max_v<int16_t,config_t::SharedScratchElementCount,1>];

// The algorithm uses this accessor to access our shared memory, used for the workgroup-level scans, also the first index is abused to broadcast single values to the whole workgroup (workgroup ticket, tile reduction, lookback index, preceding tiles' reduction)
struct ScratchAccessor
{
    // this is the example's accessor, so it's ok to assume AccessType=uint32_t accesses, but if someone uses floats for get/set it would value-convert (not bitcast) and give wrong results
    // we still need the AccessType template param, because the scan calls `scratchAccessor.template get<scalar_t,IndexType>(...)` and GenericSharedMemoryAccessor concept checks that this exists
    template<typename AccessType, typename IndexType>
    void get(const IndexType ix, NBL_REF_ARG(AccessType) value)
    {
        value = scratch[ix];
    }
    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        scratch[ix] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        spirv::controlBarrier(spv::ScopeWorkgroup, spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask);
    }
};

// The algorithm uses this accessor to write the final scanned values of its tile into the output buffer and read from the input buffer
template<uint16_t VirtualWorkgroupSize, uint16_t ItemsPerInvocation>
struct DataAccessor
{
    // function template AccessType should be the same as dtype_t
    using dtype_t = vector<uint32_t, ItemsPerInvocation>;

    static DataAccessor<VirtualWorkgroupSize, ItemsPerInvocation> create(const uint64_t inputBuf, const uint64_t outputBuf)
    {
        DataAccessor<VirtualWorkgroupSize, ItemsPerInvocation> retval;
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

// one value per workgroup tile (indexed by ticket), workgroups use it to communicate local scan results to the tiles after them
// a value is a packed `flag | (value << 2)` 
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
    void atomicStore(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = ptr + index;
        spirv::atomicStore(target.template deref().ptr.value, spv::ScopeDevice, spv::MemorySemanticsReleaseMask | spv::MemorySemanticsUniformMemoryMask, value);
    }

    void memoryBarrier()
    {
        spirv::memoryBarrier(spv::ScopeDevice, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
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

template<class Binop>
static void subtest()
{
    assert(glsl::gl_SubgroupSize() == config_t::SubgroupSize);

    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    using data_accessor_t = DataAccessor<config_t::VirtualWorkgroupSize,config_t::ItemsPerInvocation_0>;
    data_accessor_t dataAccessor = data_accessor_t::create(pc.pInputBuf, pc.pOutputBuf[Binop::BindingIndex]);

    using reduce_accessor_t = ReduceAccessor<otype_t>;
    bda::__ptr<otype_t> ptr = bda::__ptr<otype_t>::create(pc.pReduceBufs + Binop::BindingIndex * pc.reduceBufStride);
    reduce_accessor_t reduceAccessor = reduce_accessor_t::create(ptr);

    WorkgroupCounter wgCounter = WorkgroupCounter::create(pc.pWgCounterBufs + Binop::BindingIndex * pc.counterBufStride);

    ScratchAccessor scratchAccessor;
    OPERATION<config_t,binop_base_t,device_capabilities>::template __call<data_accessor_t, ScratchAccessor, reduce_accessor_t, WorkgroupCounter>(dataAccessor,scratchAccessor,reduceAccessor,wgCounter);
    // we barrier before because we alias the accessors for Binop
    scratchAccessor.workgroupExecutionAndMemoryBarrier();
}

void test()
{
    subtest<arithmetic::bit_and<uint32_t> >();
    subtest<arithmetic::bit_xor<uint32_t> >();
    subtest<arithmetic::bit_or<uint32_t> >();
    subtest<arithmetic::plus<uint32_t> >();
    subtest<arithmetic::multiplies<uint32_t> >();
    subtest<arithmetic::minimum<uint32_t> >();
    subtest<arithmetic::maximum<uint32_t> >();
}

[numthreads(config_t::WorkgroupSize,1,1)]
void main()
{
    test();
}
