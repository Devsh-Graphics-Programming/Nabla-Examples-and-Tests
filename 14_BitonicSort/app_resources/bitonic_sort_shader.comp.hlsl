#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/bitonic_sort.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using namespace nbl::hlsl;

// User-defined types for the bitonic sort
using WgType = WorkgroupType<uint32_t>;
using SortableType = pair<uint32_t, uint32_t>;  

using BitonicSortConfig = workgroup::bitonic_sort::bitonic_sort_config<
    ElementsPerThreadLog2,
    WorkgroupSizeLog2,
    WgType,
    uint32_t,               // KeyType
    SortableType,           // SortableType
    KeyComparator<uint32_t> // Comparator
>;

NBL_CONSTEXPR uint32_t WorkgroupSize = BitonicSortConfig::WorkgroupSize;

groupshared uint32_t sharedmem[BitonicSortConfig::SharedmemDWORDs];

uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(uint32_t(BitonicSortConfig::WorkgroupSize), 1, 1); }

struct ToWorkgroupType
{
    WgType operator()(SortableType sortable, uint32_t idx)
    {
        WgType wt;
        wt.key = sortable.first;
        wt.workgroupRelativeIndex = sortable.second;
        return wt;
    }
};

struct FromWorkgroupType
{
    SortableType operator()(WgType wt)
    {
        SortableType result;
        result.first = wt.key;
        result.second = wt.workgroupRelativeIndex;
        return result;
    }
};

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
	ToWorkgroupType toWgType;
	FromWorkgroupType fromWgType;

	workgroup::bitonic_sort::BitonicSort<BitonicSortConfig>::__call(accessor, sharedmemAccessor, toWgType, fromWgType);
}
