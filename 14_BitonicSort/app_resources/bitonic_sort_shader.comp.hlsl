#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/bitonic_sort.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using namespace nbl::hlsl;

using BitonicSortConfig = workgroup::bitonic_sort::bitonic_sort_config<ElementsPerThreadLog2, WorkgroupSizeLog2, uint32_t, uint32_t, less<uint32_t> >;

NBL_CONSTEXPR uint32_t WorkgroupSize = BitonicSortConfig::WorkgroupSize;

groupshared uint32_t sharedmem[BitonicSortConfig::SharedmemDWORDs];

uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(uint32_t(BitonicSortConfig::WorkgroupSize), 1, 1); }

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

struct Accessor
{
	static Accessor create(const uint64_t address)
	{
		Accessor accessor;
		accessor.address = address;
		return accessor;
	}

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

[numthreads(BitonicSortConfig::WorkgroupSize, 1, 1)]
[shader("compute")]
void main()
{
	Accessor accessor = Accessor::create(pushConstants.deviceBufferAddress);
	SharedMemoryAccessor sharedmemAccessor;

	// The sort handles load/store internally
	workgroup::BitonicSort<BitonicSortConfig>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
}