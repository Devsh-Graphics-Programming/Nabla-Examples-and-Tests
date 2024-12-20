#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using namespace nbl::hlsl;

using ConstevalParameters = workgroup::fft::ConstevalParameters<ElementsPerThreadLog2, WorkgroupSizeLog2, scalar_t>;

groupshared uint32_t sharedmem[ ConstevalParameters::SharedMemoryDWORDs];

// Users MUST define this method for FFT to work
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(uint32_t(ConstevalParameters::WorkgroupSize), 1, 1); }

struct SharedMemoryAccessor 
{
	template <typename IndexType, typename AccessType>
	void set(IndexType idx, AccessType value)
	{
		sharedmem[idx] = value;
	}

	template <typename IndexType, typename AccessType>
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

	void get(const uint32_t index, NBL_REF_ARG(complex_t<scalar_t>) value)
	{
		value = vk::RawBufferLoad<complex_t<scalar_t> >(address + index * sizeof(complex_t<scalar_t>));
	}

	void set(const uint32_t index, const complex_t<scalar_t> value)
	{
		vk::RawBufferStore<complex_t<scalar_t> >(address + index * sizeof(complex_t<scalar_t>), value);
	}

	void memoryBarrier() 
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	uint64_t address;
};

[numthreads(ConstevalParameters::WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	Accessor accessor = Accessor::create(pushConstants.deviceBufferAddress);
	SharedMemoryAccessor sharedmemAccessor;

	// FFT

	workgroup::FFT<false, ConstevalParameters>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
	workgroup::FFT<true, ConstevalParameters>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	
}