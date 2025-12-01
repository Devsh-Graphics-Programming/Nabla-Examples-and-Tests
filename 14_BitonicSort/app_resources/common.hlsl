#ifndef _BITONIC_SORT_COMMON_INCLUDED_
#define _BITONIC_SORT_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"

struct PushConstantData
{
	uint64_t deviceBufferAddress;
};

NBL_CONSTEXPR uint32_t WorkgroupSizeLog2 = 10;
NBL_CONSTEXPR uint32_t ElementsPerThreadLog2 = 2;
NBL_CONSTEXPR uint32_t elementCount = uint32_t(1) << (WorkgroupSizeLog2 + ElementsPerThreadLog2);



template<typename KeyType>
struct WorkgroupType
{
    KeyType key;
    uint32_t workgroupRelativeIndex;
};

// Packed key + subgroup index (fits in one word for efficient shuffles)
template<typename KeyType, uint32_t KeyBits, typename StorageType = uint32_t>
struct SubgroupType
{
    static const StorageType KeyMask = (StorageType(1) << KeyBits) - 1;
    StorageType packed; 

    static SubgroupType create(KeyType key, uint32_t idx)
    {
        SubgroupType st;
        st.packed = (StorageType(key) & KeyMask) | (StorageType(idx) << KeyBits);
        return st;
    }

    KeyType getKey() { return KeyType(packed & KeyMask); }
    uint32_t getIndex() { return packed >> KeyBits; }
};

template<typename K, typename Comp>
struct WorkgroupTypeComparator
{
    Comp comp;
    bool operator()(WorkgroupType<K> a, WorkgroupType<K> b) { return comp(a.key, b.key); }
};

template<typename K, uint32_t KB, typename S, typename Comp>
struct SubgroupTypeComparator
{
    Comp comp;
    bool operator()(SubgroupType<K,KB,S> a, SubgroupType<K,KB,S> b) { return comp(a.getKey(), b.getKey()); }
};


#endif
