#pragma shader_stage(compute)


#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

// static const uint32_t ArithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<ITEMS_PER_WG>::value;
// static const uint32_t BallotSz = nbl::hlsl::workgroup::scratch_size_ballot<ITEMS_PER_WG>::value;
// static const uint32_t ScratchSz = ArithmeticSz+BallotSz;

// TODO: Can we make it a static variable in the ScratchProxy struct?
// groupshared uint32_t ballotScratch[ScratchSz];  // TODO probably remove, not balloting


#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"

#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

#include "common.hlsl"

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

// #define ITEMS_PER_INVOCATION 1

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif

typedef vector<uint32_t, ITEMS_PER_INVOCATION> type_t;

// unfortunately DXC chokes on descriptors as static members
// https://github.com/microsoft/DirectXShaderCompiler/issues/5940
[[vk::binding(0, 0)]] StructuredBuffer<type_t> inputValue;
[[vk::binding(1, 0)]] RWByteAddressBuffer output[8];

// because subgroups don't match `gl_LocalInvocationIndex` snake curve addressing, we also can't load inputs that way
uint32_t globalIndex();
// since we test ITEMS_PER_WG<WorkgroupSize we need this so workgroups don't overwrite each other's outputs
bool canStore();

// #define SUBGROUP_SIZE_LOG2 5

#ifndef OPERATION
#error "Define OPERATION!"
#endif
#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif

using config_t = nbl::hlsl::workgroup2::Configuration<WORKGROUP_SIZE, SUBGROUP_SIZE_LOG2, ITEMS_PER_INVOCATION>;

groupshared vector<uint32_t, config_t::ItemsPerInvocation_1> scratch[config_t::SubgroupSize];  // final (level 1) scan needs to fit in one subgroup exactly

template<class Config>
struct ScratchProxy
{
    using stype_t = vector<uint32_t, Config::ItemsPerInvocation_1>;

    stype_t get(const uint32_t ix)
    {
        return scratch[ix];
    }
    void set(const uint32_t ix, const stype_t value)
    {
        scratch[ix] = value;
    }

    stype_t atomicOr(const uint32_t ix, const stype_t value)
    {
        return nbl::hlsl::glsl::atomicOr(scratch[ix],value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }
};

template<class Config, class Binop>
struct DataProxy
{
    using dtype_t = vector<uint32_t, Config::ItemsPerInvocation_0>;
    static_assert(nbl::hlsl::is_same_v<dtype_t, type_t>);

    dtype_t get(const uint32_t ix)
    {
        return inputValue[ix];
    }
    void set(const uint32_t ix, const dtype_t value)
    {
        // inputValue[ix] = value;
        // output[Binop::BindingIndex].template Store<type_t>(sizeof(uint32_t) + sizeof(type_t) * globalIndex(), value);
        output[Binop::BindingIndex].template Store<type_t>(sizeof(uint32_t) + sizeof(type_t) * ix, value);
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }
};

static ScratchProxy<config_t> arithmeticAccessor;

template<class Binop, class device_capabilities>
struct operation_t
{
    using binop_base_t = typename Binop::base_t;
    using otype_t = typename Binop::type_t;

    void operator()()
    {
        DataProxy<config_t,Binop> dataAccessor;
        nbl::hlsl::OPERATION<config_t,binop_base_t,device_capabilities>::template __call<DataProxy<config_t,Binop>, ScratchProxy<config_t> >(dataAccessor,arithmeticAccessor);
        // we barrier before because we alias the accessors for Binop
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
        // return retval;
    }
};


template<template<class> class binop, typename T, uint32_t N>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
    if (globalIndex()==0u)
        output[binop<T>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());
        
    operation_t<binop<T>,nbl::hlsl::jit::device_capabilities> func;
    // if (canStore())
        // output[binop<type_t>::BindingIndex].template Store<type_t>(sizeof(uint32_t)+sizeof(type_t)*globalIndex(),func());
        func(); // store is done with data accessor now
}


type_t test()
{
    const type_t sourceVal = inputValue[globalIndex()];

    subtest<bit_and, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<bit_xor, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<bit_or, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<plus, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<multiplies, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<minimum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    subtest<maximum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
    return sourceVal;
}


// template<uint16_t offset>
// struct BallotProxy
// {
//     void get(const uint32_t ix, NBL_REF_ARG(uint32_t) value)
//     {
//         value = ballotScratch[ix+offset];
//     }
//     void set(const uint32_t ix, const uint32_t value)
//     {
//         ballotScratch[ix+offset] = value;
//     }

//     uint32_t atomicOr(const uint32_t ix, const uint32_t value)
//     {
//         return nbl::hlsl::glsl::atomicOr(ballotScratch[ix],value);
//     }

//     void workgroupExecutionAndMemoryBarrier()
//     {
//         nbl::hlsl::glsl::barrier();
//         //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
//     }
// };

// static BallotProxy<ArithmeticSz> ballotAccessor;


uint32_t globalIndex()
{
    return nbl::hlsl::glsl::gl_WorkGroupID().x*ITEMS_PER_WG+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

bool canStore()
{
    return nbl::hlsl::workgroup::SubgroupContiguousIndex()<ITEMS_PER_WG;
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    const type_t sourceVal = test();
//     if (globalIndex()==0u)
//         output[ballot<type_t>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());

//     // we can only ballot booleans, so low bit
//     nbl::hlsl::workgroup::ballot<ScratchProxy<ArithmeticSz> >(bool(sourceVal & 0x1u), ballotAccessor);
//     // need to barrier between ballot and usages of a ballot by myself
//     ballotAccessor.workgroupExecutionAndMemoryBarrier();

//     uint32_t destVal = 0xdeadbeefu;
// #define CONSTEXPR_OP_TYPE_TEST(IS_OP) nbl::hlsl::is_same<nbl::hlsl::OPERATION<nbl::hlsl::bit_xor<float>,0x45>,nbl::hlsl::workgroup::IS_OP<nbl::hlsl::bit_xor<float>,0x45> >::value
// #define BALLOT_TEMPLATE_ARGS ITEMS_PER_WG,decltype(ballotAccessor),decltype(arithmeticAccessor),nbl::hlsl::jit::device_capabilities
//     if (CONSTEXPR_OP_TYPE_TEST(reduction))
//         destVal = nbl::hlsl::workgroup::ballotBitCount<BALLOT_TEMPLATE_ARGS>(ballotAccessor,arithmeticAccessor);
//     else if (CONSTEXPR_OP_TYPE_TEST(inclusive_scan))
//         destVal = nbl::hlsl::workgroup::ballotInclusiveBitCount<BALLOT_TEMPLATE_ARGS>(ballotAccessor,arithmeticAccessor);
//     else if (CONSTEXPR_OP_TYPE_TEST(exclusive_scan))
//         destVal = nbl::hlsl::workgroup::ballotExclusiveBitCount<BALLOT_TEMPLATE_ARGS>(ballotAccessor,arithmeticAccessor);
//     else
//     {
//         assert(false);
//     }
// #undef BALLOT_TEMPLATE_ARGS
// #undef CONSTEXPR_OP_TYPE_TEST

//     if (canStore())
//         output[ballot<type_t>::BindingIndex].template Store<type_t>(sizeof(uint32_t)+sizeof(type_t)*globalIndex(),destVal);
}