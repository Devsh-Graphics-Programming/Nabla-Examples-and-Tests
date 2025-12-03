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

[numthreads(ConstevalParameters::WorkgroupSize,1,1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	Accessor accessor = Accessor::create(pushConstants.deviceBufferAddress);
	SharedMemoryAccessor sharedmemAccessor;

	// FFT

	workgroup::FFT<false, ConstevalParameters>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
	workgroup::FFT<true, ConstevalParameters>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	
}