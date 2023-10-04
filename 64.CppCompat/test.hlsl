// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/mpl.hlsl>

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


[numthreads(16, 16, 1)]
void main(uint3 invocationID : SV_DispatchThreadID)
{
    {
        bool A = Spec<3>::value == 3;
    }
    {
        bool A = nbl::hlsl::type_traits::is_integral<int>::value;
    }
    int a;
    decltype(a) b = 1;
	if (all(invocationID.xy < u_pushConstants.imgSize))
	{
		// TODO use swapchain transforms
		float2 postTransformUv = float2(invocationID.xy) / float2(u_pushConstants.imgSize);
		float4 outColor = float4(postTransformUv, 0.0, 1.f);
		outImage[invocationID.xy] = outColor;
	}
}
