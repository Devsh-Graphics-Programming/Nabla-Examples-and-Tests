#include "common.hlsl"
#include "nbl/builtin/hlsl/subgroup2/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup2/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using namespace nbl::hlsl;

//using ConstevalParameters = workgroup::fft::ConstevalParameters<ElementsPerThreadLog2, WorkgroupSizeLog2, scalar_t>;

groupshared uint32_t sharedmem[4 * ((sizeof(complex_t<float32_t>) / sizeof(uint32_t)) << WorkgroupSizeLog2) ];

struct SharedMemoryAccessor 
{
	template <typename AccessType, typename IndexType>
	void set(IndexType idx, AccessType value)
	{
		sharedmem[idx] = value;
	}

	template <typename AccessType, typename IndexType>
	void get(IndexType idx, NBL_REF_ARG(AccessType) value)
	{
		value = sharedmem[idx];
	}

	void workgroupExecutionAndMemoryBarrier()
	{
		glsl::barrier();
	}

};

// Almost a LegacyBdaAccessor, but since we need `uint32_t index` getter and setter it's the same as writing one ourselves

struct Accessor
{
	static Accessor create(const uint64_t address)
    {
        Accessor accessor;
        accessor.address = address;
        return accessor;
    }

	// TODO: can't use our own BDA yet, because it doesn't support the types `workgroup::FFT` will invoke these templates with
	template <typename AccessType, typename IndexType>
	void get(const IndexType index, NBL_REF_ARG(AccessType) value)
	{
		value = vk::RawBufferLoad<AccessType>(address + index * sizeof(AccessType));
	}

	template <typename AccessType, typename IndexType>
	void set(const IndexType index, const AccessType value)
	{
		vk::RawBufferStore<AccessType>(address + index * sizeof(AccessType), value);
	}

	uint64_t address;
};


template<uint16_t Size>
struct InvocationElementsAccessor
{
	float32_t real[Size];
	float32_t imag[Size];

	void get(uint32_t channel, NBL_REF_ARG(complex_t<float32_t>) value)
	{
		value.real(real[channel]);
		value.imag(imag[channel]);
	}

	void set(uint32_t channel, complex_t<float32_t> value)
	{
		real[channel] = value.real();
		imag[channel] = value.imag();
	}
};

//[numthreads(ConstevalParameters::WorkgroupSize,1,1)]
[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	// global mem read write
	Accessor accessor = Accessor::create(pushConstants.deviceBufferAddress);
	// Load elements into the accessor
	InvocationElementsAccessor<ElementsPerThread / 2> loAcc;
	InvocationElementsAccessor<ElementsPerThread / 2> hiAcc;

	// Set up the memory adaptor
	SharedMemoryAccessor sharedmemAccessor;
	//using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, WorkgroupSize>;
	//adaptor_t sharedmemAdaptor;
	//sharedmemAdaptor.accessor = sharedmemAccessor;

	using ConstevalParameters = workgroup2::fft::ConstevalParameters<ElementsPerThread, SubgroupSizeLog2, WorkgroupSizeLog2, ShuffledChannelsPerRound, false, 0, ShareTwiddles, float32_t>;
	using FFT = workgroup2::FFT<false, ConstevalParameters>;
	using IFFT = workgroup2::FFT<true, ConstevalParameters>;

	// Invert last channel to ensure ping pong works
	[unroll]
	for (uint32_t pair = 0u; pair < ElementsPerThread / 2; pair++)
	{
		complex_t<float32_t> lo, hi;
		accessor.get(uint32_t(workgroup::SubgroupContiguousIndex()) + 2 * pair * WorkgroupSize, lo);
		loAcc.set(pair, lo);
		accessor.get(uint32_t(workgroup::SubgroupContiguousIndex()) + (2 * pair + 1) * WorkgroupSize, hi);
		hiAcc.set(pair, hi);
		//printf("Pair %d is lo: %f, %f hi: %f, %f", pair, lo.real(), lo.imag(), hi.real(), hi.imag());
		//printf("SharedmemSize: %d", 4 * ((sizeof(complex_t<float32_t>) / sizeof(uint32_t)) << WorkgroupSizeLog2));
		//printf("ShuffleRounds: %d", ConstevalParameters::ShuffleRounds);
	}

	FFT::__call(loAcc, hiAcc, sharedmemAccessor);
	sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
	IFFT::__call(loAcc, hiAcc, sharedmemAccessor);

	[unroll]
	for (uint32_t pair = 0u; pair < ElementsPerThread / 2; pair++)
	{
		complex_t<float32_t> lo, hi;
		loAcc.get(pair, lo);
		accessor.set(uint32_t(workgroup::SubgroupContiguousIndex()) + 2 * pair * WorkgroupSize, lo);
		hiAcc.get(pair, hi);
		accessor.set(uint32_t(workgroup::SubgroupContiguousIndex()) + (2 * pair + 1) * WorkgroupSize, hi);
	}
}