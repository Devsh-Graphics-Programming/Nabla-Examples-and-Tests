#include "nbl/builtin/hlsl/central_limit_blur/common.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

[[vk::binding( 0, 0 )]] Texture2D<nbl::hlsl::float32_t4> input;
[[vk::binding( 1, 0 )]] RWTexture2D<nbl::hlsl::float32_t4> output;

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
		}
	}

	void poststore( uint16_t axisIdx, uint16_t itemIdx )
	{
		uint32_t2 texSize;
		output.GetDimensions( texSize.x, texSize.y );

		uint32_t2 coordinate = textureCoord( axisIdx, itemIdx );
		if( all( coordinate < texSize ) )
		{
			output[ coordinate ] = localSpill[ itemIdx ];
		}
	}

	nbl::hlsl::float32_t4 get( uint16_t itemIdx, uint16_t ch )
	{
		return localSpill[ itemIdx ][ ch ];
	}

	void set( uint16_t itemIdx, uint16_t ch, NBL_CONST_REF_ARG( float32_t ) val )
	{
		localSpill[ itemIdx ][ ch ] = val;
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