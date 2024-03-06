#include "nbl/builtin/hlsl/central_limit_blur/common.hlsl"

[[vk::binding( 0, 0 )]] Texture2D<nbl::hlsl::float32_t4> input;
[[vk::binding( 1, 0 )]] RWTexture2D<nbl::hlsl::float32_t4> output;


struct TextureProxy
{
	nbl::hlsl::float32_t4 get( const uint32_t2 coordinate )
	{
		uint32_t3 texSize;
		input.GetDimensions( 0, texSize.x, texSize.y, texSize.z );

		float32_t4 data = float32_t4(0.f, 0.f, 0.f, 0.f);
		if( all( coordinate < texSize.xy ) )
		{
			float32_t4 pixel = input[ coordinate ];
			data = pixel;// [channel] ;
		}

		return data;
	}

	void set( const uint32_t2 coordinate, NBL_CONST_REF_ARG(float32_t4) val )
	{
		uint32_t2 texSize;
		output.GetDimensions( texSize.x, texSize.y );

		if( all( coordinate < texSize ) )
		{
			output[ coordinate ] = val;
		}
	}
};
