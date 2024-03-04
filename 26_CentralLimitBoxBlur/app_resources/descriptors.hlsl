#include "common.hlsl"

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
