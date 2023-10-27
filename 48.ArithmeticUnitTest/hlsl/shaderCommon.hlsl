#pragma shader_stage(compute)

#include "common.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

// annoying things necessary to do until DXC implements proposal 0011
static uint32_t __gl_LocalInvocationIndex;
uint32_t nbl::hlsl::glsl::gl_LocalInvocationIndex() {return __gl_LocalInvocationIndex;}
static uint32_t3 __gl_GlobalInvocationID;
uint32_t3 nbl::hlsl::glsl::gl_GlobalInvocationID() {return __gl_GlobalInvocationID;}



// TODO: see if DXC chokes if we make these static :D
[[vk::binding(0, 0)]] StructuredBuffer<uint> inputValue; // read-only

[[vk::binding(1, 0)]] RWStructuredBuffer<Output> outand;
[[vk::binding(2, 0)]] RWStructuredBuffer<Output> outxor;
[[vk::binding(3, 0)]] RWStructuredBuffer<Output> outor;
[[vk::binding(4, 0)]] RWStructuredBuffer<Output> outadd;
[[vk::binding(5, 0)]] RWStructuredBuffer<Output> outmul;
[[vk::binding(6, 0)]] RWStructuredBuffer<Output> outmin;
[[vk::binding(7, 0)]] RWStructuredBuffer<Output> outmax;
[[vk::binding(8, 0)]] RWStructuredBuffer<Output> outbitcount;

template<template<class> class operation_t>
struct test
{
	static void run()
	{
		const uint32_t globalIx = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
		const uint32_t sourceVal = inputValue[globalIx];

		outand[0].subgroupSize
			= outxor[0].subgroupSize
			= outor[0].subgroupSize
			= outadd[0].subgroupSize
			= outmul[0].subgroupSize
			= outmin[0].subgroupSize
			= outmax[0].subgroupSize
			= nbl::hlsl::glsl::gl_SubgroupSize();

		operation_t<nbl::hlsl::bit_and<nbl::hlsl::remove_const<decltype(sourceVal)>::type> > r_and;
		outand[0].output[globalIx] = r_and(sourceVal);

		operation_t<nbl::hlsl::bit_xor<nbl::hlsl::remove_const<decltype(sourceVal)>::type> > r_xor;
		outxor[0].output[globalIx] = r_xor(sourceVal);

		operation_t<nbl::hlsl::bit_or<nbl::hlsl::remove_const<decltype(sourceVal)>::type> > r_or;
		outor[0].output[globalIx] = r_or(sourceVal);

		operation_t<nbl::hlsl::plus<nbl::hlsl::remove_const<decltype(sourceVal)>::type> > r_add;
		outadd[0].output[globalIx] = r_add(sourceVal);

		operation_t<nbl::hlsl::multiplies<nbl::hlsl::remove_const<decltype(sourceVal)>::type> > r_mul;
		outmul[0].output[globalIx] = r_mul(sourceVal);

		operation_t<nbl::hlsl::minimum<nbl::hlsl::remove_const<decltype(sourceVal)>::type> > r_min;
		outmin[0].output[globalIx] = r_min(sourceVal);

		operation_t<nbl::hlsl::maximum<nbl::hlsl::remove_const<decltype(sourceVal)>::type> > r_max;
		outmax[0].output[globalIx] = r_max(sourceVal);
	}
};