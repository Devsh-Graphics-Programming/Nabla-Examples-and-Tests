// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

struct PushConstants
{
	uint2 imgSize;
	uint swapchainTransform;
};

[[vk::push_constant]]
PushConstants u_pushConstants;

[[vk::binding(0, 0)]] RWTexture2D<float4> outImage;
[[vk::binding(1, 0)]] Texture2D<float4> inImage;

[numthreads(16, 16)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID)
{
	if (all(gl_GlobalInvocationID.xy < u_pushConstants.imgSize))
	{
		// TODO use swapchain transforms
		float2 postTransformUv = float2(gl_GlobalInvocationID.xy) / float2(u_pushConstants.imgSize);
		float4 outColor = float4(postTransformUv, 0.0, 1.f);
		outImage[gl_GlobalInvocationID.xy] = outColor;
	}
}
