// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/central_limit_blur/common.hlsl"
#include "descriptors.hlsl"

#include "nbl/builtin/hlsl/central_limit_blur/box_blur.hlsl"

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
nbl::hlsl::uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() { return uint32_t3( WORKGROUP_SIZE, 1, 1 ); }

static const uint32_t ITEMS_PER_THREAD = 4; // ?

static const uint32_t arithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value;
static const uint32_t smemSize = WORKGROUP_SIZE + arithmeticSz;

groupshared uint32_t scratch[ smemSize ];

template<typename T, uint16_t offset>
struct ScratchProxy
{
	T get( const uint32_t ix )
	{
		return nbl::hlsl::bit_cast< T, uint32_t >( scratch[ ix + offset ] );
	}
	void set( const uint32_t ix, NBL_CONST_REF_ARG( T ) value )
	{
		scratch[ ix + offset ] = nbl::hlsl::bit_cast< uint32_t, T >( value );
	}

	void workgroupExecutionAndMemoryBarrier()
	{
		nbl::hlsl::glsl::barrier();
	}
};

[[vk::push_constant]]
nbl::hlsl::central_limit_blur::BoxBlurParams boxBlurParams;

[numthreads( WORKGROUP_SIZE, 1, 1 )]
void main( uint3 invocationID : SV_DispatchThreadID )
{
	uint16_t axisIdx = uint16_t( boxBlurParams.direction );
	uint32_t wrapMode = boxBlurParams.wrapMode;
	nbl::hlsl::float32_t4 borderColor = boxBlurParams.getBorderColor();

	ScratchProxy<float32_t, 0> scratchProxy;
	ScratchProxy<float32_t, WORKGROUP_SIZE> arithmeticProxy;

	TextureProxy<ITEMS_PER_THREAD> textureProxy;
	for( uint16_t i = 0u; i < ITEMS_PER_THREAD; ++i )
	{
		textureProxy.preload( axisIdx, i );
	}
	nbl::hlsl::glsl::barrier();

	for( uint16_t chIdx = 0u; chIdx < boxBlurParams.channelCount; ++chIdx )
	{
		nbl::hlsl::central_limit_blur::BoxBlur<__decltype(textureProxy), __decltype( scratchProxy ), __decltype( arithmeticProxy ), WORKGROUP_SIZE, arithmeticSz>( 1, ITEMS_PER_THREAD, chIdx, boxBlurParams.radius, wrapMode, borderColor, textureProxy, scratchProxy, arithmeticProxy );
	}

	nbl::hlsl::glsl::barrier();
	for( uint16_t i = 0; i < ITEMS_PER_THREAD; ++i )
	{
		textureProxy.poststore( axisIdx, i );
	}
	
}
