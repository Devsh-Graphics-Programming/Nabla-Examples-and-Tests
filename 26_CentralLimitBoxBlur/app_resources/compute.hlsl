// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma shader_stage(compute)

#include "common.hlsl"
#include "box_blur.hlsl"

[[vk::push_constant]]
BBoxBlurParams boxBlurParams;

[numthreads( DefaultWorkgroupSize, 1, 1 )]
void main( uint3 invocationID : SV_DispatchThreadID )
{
	uint32_t direction = boxBlurParams.getDirection();
	uint32_t wrapMode = boxBlurParams.getWrapMode();
	nbl::hlsl::float32_t4 borderColor = float32_t4(1.f, 0.f, 1.f, 1.f);
	if( boxBlurParams.getWrapMode() == WRAP_MODE_CLAMP_TO_BORDER )
	{
		borderColor = boxBlurParams.getBorderColor();
	}

	BufferAccessor textureAccessor = BufferAccessorCtor( boxBlurParams.inputDimensions, boxBlurParams.inputStrides,
														 boxBlurParams.outputStrides );

	for( uint32_t ch = 0; ch < boxBlurParams.getChannelCount(); ++ch )
	{
		BoxBlur( ch, direction, boxBlurParams.radius, wrapMode, borderColor, textureAccessor );
	}
}
