#include "common.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

// annoying things necessary to do until DXC implements proposal 0011
static uint32_t __gl_LocalInvocationIndex;
uint32_t nbl::hlsl::glsl::gl_LocalInvocationIndex() {return __gl_LocalInvocationIndex;}
static uint32_t3 __gl_GlobalInvocationID;
uint32_t3 nbl::hlsl::glsl::gl_GlobalInvocationID() {return __gl_GlobalInvocationID;}


// unfortunately DXC chokes on descriptors as static members
// https://github.com/microsoft/DirectXShaderCompiler/issues/5940
[[vk::binding(0, 0)]] StructuredBuffer<uint32_t> inputValue;
[[vk::binding(1, 0)]] RWByteAddressBuffer output[8];


#ifndef OPERATION
#error "Define OPERATION!"
#endif
template<template<class> class binop, typename T>
static void subtest(NBL_CONST_REF_ARG(T) sourceVal)
{
	const uint32_t globalIndex = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
	if (globalIndex==0u)
		output[binop<T>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());
		
	operation_t<typename binop<T>::base_t> func;
	output[binop<T>::BindingIndex].template Store<T>(sizeof(uint32_t)+sizeof(T)*globalIndex,func(sourceVal));
}

uint32_t test()
{
	const uint32_t sourceVal = inputValue[nbl::hlsl::glsl::gl_GlobalInvocationID().x];

	subtest<bit_and>(sourceVal);
	subtest<bit_xor>(sourceVal);
	subtest<bit_or>(sourceVal);
	subtest<plus>(sourceVal);
	subtest<multiplies>(sourceVal);
	subtest<minimum>(sourceVal);
	subtest<maximum>(sourceVal);
	return sourceVal;
}