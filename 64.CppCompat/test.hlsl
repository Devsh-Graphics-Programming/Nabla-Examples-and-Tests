// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

#include <nbl/builtin/hlsl/type_traits.hlsl>

#include <nbl/builtin/hlsl/mpl.hlsl>

#include <nbl/builtin/hlsl/limits.hlsl>

struct PushConstants
{
	uint2 imgSize;
	uint swapchainTransform;
};

[[vk::push_constant]]
PushConstants u_pushConstants;

[[vk::binding(0, 0)]] RWTexture2D<float4> outImage;
[[vk::binding(1, 0)]] Texture2D<float4> inImage;


template<int A>
struct Spec
{
    static const int value = Spec<A-1>::value + 1;
};

template<>
struct Spec<0>
{
    static const int value = 0;
};


Buffer<float32_t4>  unbounded[];

template<class T>
bool val(T) { return nbl::hlsl::is_unbounded_array<T>::value; }



[numthreads(16, 16, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    {
        bool A = Spec<3>::value == 3;
    }
    {
        bool A = nbl::hlsl::is_integral<int>::value;
    }
    {
        bool A = val(unbounded);
    }

    {
        int Q[3][4][5];
        nbl::hlsl::conditional<3 == nbl::hlsl::extent<__decltype(Q), 0>::value, int, void>::type a = 0;
        nbl::hlsl::conditional<4 == nbl::hlsl::extent<__decltype(Q), 1>::value, int, void>::type b = 0;
        nbl::hlsl::conditional<5 == nbl::hlsl::extent<__decltype(Q), 2>::value, int, void>::type c = 0;
        nbl::hlsl::conditional<0 == nbl::hlsl::extent<__decltype(Q), 3>::value, int, void>::type d = 0;
    }

	if (all(invocationID.xy < u_pushConstants.imgSize))
	{
		// TODO use swapchain transforms
		float2 postTransformUv = float2(invocationID.xy) / float2(u_pushConstants.imgSize);
		float4 outColor = float4(postTransformUv, 0.0, 1.f);
		outImage[invocationID.xy] = outColor;
	}
}
