#include "nbl/builtin/hlsl/central_limit_blur/common.hlsl"

[[vk::binding( 0, 0 )]] Texture2D<nbl::hlsl::float32_t4> input;
[[vk::binding( 1, 0 )]] RWTexture2D<nbl::hlsl::float32_t4> output;


struct BufferAccessor
{
	uint32_t2 chosenAxis;
	
	nbl::hlsl::float32_t get( const uint32_t linearIndex, const uint32_t channel )
	{
		uint32_t3 texSize;
		input.GetDimensions( 0, texSize.x, texSize.y, texSize.z );

		uint32_t axisSize = dot( texSize.xy, chosenAxis );

		uint32_t2 coordinate = { linearIndex % axisSize, linearIndex / axisSize };
		float32_t data = 0.f;
		if( all( coordinate < texSize.xy ) )
		{
			float32_t4 pixel = input[ coordinate.xy ];
			data = pixel[ channel ];
		}

		return data;
	}

	void set( const uint32_t linearIndex, const uint32_t channel, NBL_CONST_REF_ARG( float32_t ) val )
	{
		uint32_t2 texSize;
		output.GetDimensions( texSize.x, texSize.y );

		uint32_t axisSize = dot( texSize, chosenAxis );

		uint32_t2 coordinate = { linearIndex % axisSize, linearIndex / axisSize };
		if( all( coordinate < texSize ) )
		{
			float32_t4 col = float32_t4( 0.f, 0.f, 0.f, 0.f );
			col[ channel ] = val;
			output[ coordinate.xy ] = col;
		}
	}
};

BufferAccessor BufferAccessorCtor( uint32_t2 chosenAxis )
{
	BufferAccessor ba;
	ba.chosenAxis = chosenAxis;
	return ba;
}
