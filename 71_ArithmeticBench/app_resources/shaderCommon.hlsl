#include "common.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

// unfortunately DXC chokes on descriptors as static members
// https://github.com/microsoft/DirectXShaderCompiler/issues/5940
[[vk::binding(0, 0)]] StructuredBuffer<uint32_t> inputValue;
[[vk::binding(1, 0)]] RWByteAddressBuffer output[8];

// because subgroups don't match `gl_LocalInvocationIndex` snake curve addressing, we also can't load inputs that way
uint32_t globalIndex();
// since we test ITEMS_PER_WG<WorkgroupSize we need this so workgroups don't overwrite each other's outputs
bool canStore();

#ifndef ITEMS_PER_INVOCATION
#error "Define ITEMS_PER_INVOCATION!"
#endif
//typedef decltype(inputValue[0]) type_t;
//typedef uint32_t type_t;
//typedef uint32_t4 type_t;
typedef vector<uint32_t, ITEMS_PER_INVOCATION> type_t;


#ifndef OPERATION
#error "Define OPERATION!"
#endif
// template<template<class> class binop>
// static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
// {
// 	if (globalIndex()==0u)
// 		output[binop::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());
		
// 	operation_t<typename binop<type_t>::base_t,nbl::hlsl::jit::device_capabilities> func;
// 	if (canStore())
// 		output[binop::BindingIndex].template Store<type_t>(sizeof(uint32_t)+sizeof(type_t)*globalIndex(),func(sourceVal));
// }

#ifndef SUBGROUP_SIZE_LOG2
#error "Define SUBGROUP_SIZE_LOG2!"
#endif
template<template<class> class binop, typename T, uint32_t N>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
	// TODO static assert vector<T, N> == type_t
	//using type_t = vector<T, N>;
	using config_t = nbl::hlsl::subgroup::Configuration<SUBGROUP_SIZE_LOG2>;
	using params_t = nbl::hlsl::subgroup2::ArithmeticParams<config_t, typename binop<T>::base_t, N, nbl::hlsl::jit::device_capabilities>;

	if (globalIndex()==0u)
		output[binop<T>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());
		
	operation_t<params_t> func;
	if (canStore())
		output[binop<T>::BindingIndex].template Store<type_t>(sizeof(uint32_t)+sizeof(type_t)*globalIndex(),func(sourceVal));
}


type_t test()
{
	const uint32_t idx = globalIndex() * ITEMS_PER_INVOCATION;
	type_t sourceVal;
	[unroll]
	for (uint32_t i = 0; i < ITEMS_PER_INVOCATION; i++)
	{
		sourceVal[i] = inputValue[idx + i];
	}

	subtest<bit_and, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
	subtest<bit_xor, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
	subtest<bit_or, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
	subtest<plus, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
	subtest<multiplies, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
	subtest<minimum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
	subtest<maximum, uint32_t, ITEMS_PER_INVOCATION>(sourceVal);
	return sourceVal;
}

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
