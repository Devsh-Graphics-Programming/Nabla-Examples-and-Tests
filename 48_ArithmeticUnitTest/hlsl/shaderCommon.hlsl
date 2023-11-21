#include "common.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

// annoying things necessary to do until DXC implements proposal 0011
static uint32_t __gl_LocalInvocationIndex;
uint32_t nbl::hlsl::glsl::gl_LocalInvocationIndex() {return __gl_LocalInvocationIndex;} // need this becacuse of the circular way `gl_SubgroupID` is currently defined
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}
static uint32_t3 __gl_WorkGroupID;
uint32_t3 nbl::hlsl::glsl::gl_WorkGroupID() {return __gl_WorkGroupID;}


// unfortunately DXC chokes on descriptors as static members
// https://github.com/microsoft/DirectXShaderCompiler/issues/5940
[[vk::binding(0, 0)]] StructuredBuffer<uint32_t> inputValue;
[[vk::binding(1, 0)]] RWByteAddressBuffer output[8];

// because subgroups don't match `gl_LocalInvocationIndex` snake curve addressing, we also can't load inputs that way
uint32_t globalIndex();
// since we test ITEMS_PER_WG<WorkgroupSize we need this so workgroups don't overwrite each other's outputs
bool canStore();

//typedef decltype(inputValue[0]) type_t;
typedef uint32_t type_t;


#ifndef OPERATION
#error "Define OPERATION!"
#endif
template<template<class> class binop>
static void subtest(NBL_CONST_REF_ARG(type_t) sourceVal)
{
	if (globalIndex()==0u)
		output[binop<type_t>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());
		
	operation_t<typename binop<type_t>::base_t> func;
	if (canStore())
		output[binop<type_t>::BindingIndex].template Store<type_t>(sizeof(uint32_t)+sizeof(type_t)*globalIndex(),func(sourceVal));
}


type_t test()
{
	const type_t sourceVal = inputValue[globalIndex()];

	subtest<bit_and>(sourceVal);
	subtest<bit_xor>(sourceVal);
	subtest<bit_or>(sourceVal);
	subtest<plus>(sourceVal);
	subtest<multiplies>(sourceVal);
	subtest<minimum>(sourceVal);
	subtest<maximum>(sourceVal);
	return sourceVal;
}

#include "nbl/builtin/hlsl/workgroup/basic.hlsl"