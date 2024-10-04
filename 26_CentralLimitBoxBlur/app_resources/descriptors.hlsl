#pragma once

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

NBL_CONSTEXPR_STATIC_INLINE uint32_t inputViewBinding = 0u;
NBL_CONSTEXPR_STATIC_INLINE uint32_t outputViewBinding = 1u;


#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/central_limit_blur/common.hlsl>
#include <nbl/builtin/hlsl/workgroup/basic.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/colorspace/OETF.hlsl>

// [[vk::binding( inputViewBinding, 0 )]] Texture2D<nbl::hlsl::float32_t4> input;
[[vk::binding( inputViewBinding, 0 )]] Texture2D<nbl::hlsl::float32_t4> input;
[[vk::binding( outputViewBinding, 0 )]] RWTexture2D<nbl::hlsl::float32_t4> output;

template<uint16_t ITEMS_PER_THREAD>
struct TextureProxy
{
	void preload( uint16_t axisIdx, uint16_t itemIdx )
	{
		uint32_t3 texSize;
		input.GetDimensions( 0, texSize.x, texSize.y, texSize.z );

		float32_t4 data = float32_t4(0.f, 0.f, 0.f, 0.f);

		uint32_t2 coordinate = textureCoord( axisIdx, itemIdx );
		if( all( coordinate < texSize.xy ) )
		{
			float32_t4 pixel = input[ coordinate ];
			localSpill[ itemIdx ] = pixel;
			//localSpill[ itemIdx ] = float32_t4( 1.f, 1.f, 1.f, 1.f );
		}
	}

	void poststore( uint16_t axisIdx, uint16_t itemIdx )
	{
		uint32_t2 texSize;
		output.GetDimensions( texSize.x, texSize.y );

		uint32_t2 coordinate = textureCoord( axisIdx, itemIdx );
		if( all( coordinate < texSize ) )
		{
			float32_t4 outCol = float32_t4( nbl::hlsl::colorspace::oetf::sRGB<float32_t3>( localSpill[ itemIdx ].xyz ), localSpill[ itemIdx ].w );
			output[ coordinate ] = outCol;
			// output[ coordinate ] = localSpill[ itemIdx ];
		}
	}

	nbl::hlsl::float32_t get( uint16_t itemIdx, uint16_t ch )
	{
		float32_t4 thisSpill = localSpill[ itemIdx ];
		return thisSpill[ ch ];
	}

	void set( uint16_t itemIdx, uint16_t ch, NBL_CONST_REF_ARG(float32_t) val )
	{
		float32_t4 thisSpill = localSpill[ itemIdx ];
		thisSpill[ ch ] = val;
		localSpill[ itemIdx ] = thisSpill;
	}

//private:
	float32_t4 localSpill[ ITEMS_PER_THREAD ];

	uint32_t2 textureCoord( uint16_t axisIdx, uint16_t idx )
	{
		uint32_t3 coord = nbl::hlsl::glsl::gl_WorkGroupID();
		coord[ axisIdx ] = ( idx * nbl::hlsl::glsl::gl_WorkGroupSize().x ) + nbl::hlsl::workgroup::SubgroupContiguousIndex();
		return coord.xy;
	}
};

#endif