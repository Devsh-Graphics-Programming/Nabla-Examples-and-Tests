// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/blur/common.hlsl"

//#include "descriptors"
////////////////////////////
[[vk::binding( 0, 0 )]] Buffer<nbl::hlsl::float32_t> input;
[[vk::binding( 1, 0 )]] RWBuffer<nbl::hlsl::float32_t> output;


// TODO: figure the proper way to do templated BufferAccessors
struct BufferAccessor
{
	uint32_t3 dimension;
	uint32_t inputStride;
	uint32_t outputStride;
	//uint32_t channelCount;

	nbl::hlsl::float32_t getPaddedData( const uint32_t3 coordinate, const uint32_t channel )
	{
		uint32_t stridedIdx = dot( uint32_t4( coordinate, channel ), inputStride );

		float data = 0.f;
		if( all( coordinate < dimension ) )
		{
			data = input[ stridedIdx ];
		}

		return data;
	}

	void setData( const uint32_t3 coordinate, const uint32_t channel, const float32_t val )
	{
		if( all( coordinate < dimension ) )
		{
			uint32_t strided_idx = dot( uint32_t4( coordinate, channel ), outputStride );
			output[ strided_idx ] = val;
		}
	}
};

BufferAccessor BufferAccessorCtor( uint32_t3 dimension, uint32_t inputStride, uint32_t outputStride )
{
	BufferAccessor ba;
	ba.dimension = dimension;
	ba.inputStride = inputStride;
	ba.outputStride = outputStride;
	return ba;
}
////////////////////////////

#include "nbl/builtin/hlsl/blur/box_blur.hlsl"

[[vk::push_constant]]
BoxBlurParams boxBlurParams;

[numthreads( WORKGROUP_SIZE, 1, 1 )]
void main( uint3 invocationID : SV_DispatchThreadID )
{
	uint32_t direction = boxBlurParams.getDirection();
	uint32_t wrapMode = boxBlurParams.getWrapMode();
	nbl::hlsl::float32_t4 borderColor = float32_t4(1.f, 0.f, 1.f, 1.f);
	if( boxBlurParams.getWrapMode() == WRAP_MODE_CLAMP_TO_BORDER )
	{
		borderColor = boxBlurParams.getBorderColor();
	}

	BufferAccessor textureAccessor = BufferAccessorCtor( 
		boxBlurParams.inputDimensions.xyz, boxBlurParams.inputStrides, boxBlurParams.outputStrides );

	for( uint32_t ch = 0; ch < boxBlurParams.getChannelCount(); ++ch )
	{
		BoxBlur( ch, direction, boxBlurParams.radius, wrapMode, borderColor, textureAccessor );
	}
}
