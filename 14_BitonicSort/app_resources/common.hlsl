#ifndef _BITONIC_SORT_COMMON_INCLUDED_
#define _BITONIC_SORT_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
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

template<typename KeyType, uint32_t KeyBits, typename StorageType = uint32_t>
struct SubgroupType
{
    static const StorageType KeyMask = (StorageType(1) << KeyBits) - 1;
    StorageType packed;

    static SubgroupType create(KeyType key, uint32_t subgroupRelativeIndex)
    {
        SubgroupType st;
        st.packed = (StorageType(key) & KeyMask) | (StorageType(subgroupRelativeIndex) << KeyBits);
        return st;
    }

    KeyType getKey() { return KeyType(packed & KeyMask); }
    uint32_t getSubgroupRelativeIndex() { return packed >> KeyBits; }

};

template<typename KeyType>
struct KeyComparator
{
    bool operator()(KeyType a, KeyType b)
    {
        return a < b;
    }

    bool operator()(nbl::hlsl::pair<KeyType, KeyType> a, nbl::hlsl::pair<KeyType, KeyType> b)
    {
        return a.first < b.first; 
    }

    bool operator()(WorkgroupType<KeyType> a, WorkgroupType<KeyType> b)
    {
        return a.key < b.key;
    }

    template<uint32_t KeyBits, typename StorageType>
    bool operator()(SubgroupType<KeyType, KeyBits, StorageType> a,
                    SubgroupType<KeyType, KeyBits, StorageType> b)
    {
        return a.getKey() < b.getKey();
    }
};

#endif
