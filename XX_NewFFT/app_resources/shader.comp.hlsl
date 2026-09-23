#include "common.hlsl"
#include "nbl/builtin/hlsl/subgroup2/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup2/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using namespace nbl::hlsl;

groupshared uint32_t sharedmem[4 * ((sizeof(complex_t<scalar_t>) / sizeof(uint32_t)) << WorkgroupSizeLog2) ];

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


template<uint16_t Channels, uint16_t Size>
struct InvocationElementsAccessor
{
	scalar_t real[Channels][Size];
	scalar_t imag[Channels][Size];

	void get(uint32_t channel, uint32_t pair, NBL_REF_ARG(complex_t<scalar_t>) value)
	{
		value.real(real[channel][pair]);
		value.imag(imag[channel][pair]);
	}

	void set(uint32_t channel, uint32_t pair, NBL_CONST_REF_ARG(complex_t<scalar_t>) value)
	{
		real[channel][pair] = value.real();
		imag[channel][pair] = value.imag();
	}
};

using _InvocationElementsAccessor = InvocationElementsAccessor<Channels, ElementsPerInvocationPerChannel / 2>;
using ElementsAccessorAdaptor = workgroup2::fft::WorkgroupRadix2AccessorAdaptor<Channels, scalar_t, _InvocationElementsAccessor>;

//[numthreads(ConstevalParameters::WorkgroupSize,1,1)]
[numthreads(WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	// global mem read write
	Accessor accessor = Accessor::create(pushConstants.deviceBufferAddress);
	// Load elements into the accessor
	_InvocationElementsAccessor loElementAccessor;
	ElementsAccessorAdaptor loAcc = ElementsAccessorAdaptor::create(loElementAccessor);
	_InvocationElementsAccessor hiElementAccessor;
	ElementsAccessorAdaptor hiAcc = ElementsAccessorAdaptor::create(hiElementAccessor);

	// Set up the memory adaptor
	SharedMemoryAccessor sharedmemAccessor;
	//using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, WorkgroupSize>;
	//adaptor_t sharedmemAdaptor;
	//sharedmemAdaptor.accessor = sharedmemAccessor;

	using FFT = workgroup2::impl::InnerFFT<false, ConstevalParametersForward>;
	using IFFT = workgroup2::impl::InnerFFT<true, ConstevalParametersInverse>;

	// Invert last channel to ensure ping pong works
	[unroll]
	for (uint32_t pair = 0u; pair < ElementsPerInvocationPerChannel / 2; pair++)
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
	for (uint32_t pair = 0u; pair < ElementsPerInvocationPerChannel / 2; pair++)
	{
		complex_t<float32_t> lo, hi;
		loAcc.get(pair, lo);
		accessor.set(uint32_t(workgroup::SubgroupContiguousIndex()) + 2 * pair * WorkgroupSize, lo);
		hiAcc.get(pair, hi);
		accessor.set(uint32_t(workgroup::SubgroupContiguousIndex()) + (2 * pair + 1) * WorkgroupSize, hi);
	}
}