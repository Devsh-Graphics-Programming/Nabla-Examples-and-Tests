#include "nbl/builtin/hlsl/blur/common.hlsl"

[[vk::binding( 0, 0 )]] Texture2D<nbl::hlsl::float32_t> input;
[[vk::binding( 1, 0 )]] RWTexture2D<nbl::hlsl::float32_t> output;


// TODO: figure the proper way to do templated BufferAccessors
struct BufferAccessor
{
	uint32_t4 inputStride;
	uint32_t4 outputStride;
	uint32_t3 dimension;
	//uint32_t channelCount;
	// mod image width x div image width y 
	nbl::hlsl::float32_t getPaddedData( const uint32_t3 coordinate, const uint32_t channel )
	{
		float data = 0.f;
		if( all( coordinate < dimension ) )
		{
			uint32_t stridedIdx = dot( uint32_t4( coordinate, channel ), inputStride );// NOT CORRECT
			//uint32_t2 idx = stridedIdx % 
			//data = input[ stridedIdx ];
		}

		return data;
	}

	void setData( const uint32_t3 coordinate, const uint32_t channel, NBL_CONST_REF_ARG( float32_t ) val )
	{
		if( all( coordinate < dimension ) )
		{
			uint32_t stridedIdx = dot( uint32_t4( coordinate, channel ), outputStride ); // NOT CORRECT
			//output[ stridedIdx ] = val;
		}
	}
};

BufferAccessor BufferAccessorCtor( uint32_t4 inputStride, uint32_t4 outputStride, uint32_t3 dimension )
{
	BufferAccessor ba;
	ba.dimension = dimension;
	ba.inputStride = inputStride;
	ba.outputStride = outputStride;
	return ba;
}
