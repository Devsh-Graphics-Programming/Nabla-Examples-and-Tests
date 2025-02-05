#ifndef _NBL_HLSL_MPMC_QUEUE_HLSL_
#define _NBL_HLSL_MPMC_QUEUE_HLSL_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"

namespace nbl
{
namespace hlsl
{
    
// ONLY ONE INVOCATION IN A WORKGROUP USES THESE
// NO OVERFLOW PROTECTION, YOU MUSTN'T OVERFLOW!
template<typename T>
struct MPMCQueue
{
    const static uint64_t ReservedOffset = 0;
    const static uint64_t ComittedOffset = 1;
    const static uint64_t PoppedOffset = 2;

    // we don't actually need to use 64-bit offsets/counters for the 
    uint64_t getStorage(const uint32_t ix)
    {
        return pStorage+(ix&((0x1u<<capacityLog2)-1))*sizeof(T);
    }

    void push(const in T val)
    {
        // reserve output index
        bda::__ref<uint32_t,4> reserved = (counters+ReservedOffset).deref();
        const uint32_t dstIx = spirv::atomicIAdd(reserved.__get_spv_ptr(),spv::ScopeWorkgroup,spv::MemorySemanticsAcquireMask,1);
        // write
        vk::RawBufferStore(getStorage(dstIx),val);
        // say its ready for consumption
        bda::__ref<uint32_t,4> committed = (counters+ComittedOffset).deref();
        spirv::atomicUMax(committed.__get_spv_ptr(),spv::ScopeWorkgroup,spv::MemorySemanticsReleaseMask,1);
    }

    // everything here is must be done by one invocation between two barriers, all invocations must call this method
    // `electedInvocation` must true only for one invocation and be such that `endOffsetInPopped` has the highest value amongst them
    template<typename BroadcastAccessor>
    bool pop(BroadcastAccessor accessor, const bool active, out T val, const uint16_t endOffsetInPopped, const bool electedInvocation, const uint32_t beginHint)
    {
        if (electedInvocation)
        {
            uint32_t begin;
            uint32_t end;
            // strictly speaking CAS loops have FP because one WG will perform the comp-swap and make progress
            uint32_t expected;
            bda::__ref<uint32_t,4> committed = (counters+ComittedOffset).deref();
            bda::__ref<uint32_t,4> popped = (counters+PoppedOffset).deref();
            do
            {
                // TOOD: replace `atomicIAdd(p,0)` with `atomicLoad(p)`
                uint32_t end = spirv::atomicIAdd(committed.__get_spv_ptr(),spv::ScopeWorkgroup,spv::MemorySemanticsAcquireReleaseMask,0u);
                end = min(end,begin+endOffsetInPopped);
                expected = begin;
                begin = spirv::atomicCompareExchange(
                    popped.__get_spv_ptr(),
                    spv::ScopeWorkgroup,
                    spv::MemorySemanticsAcquireReleaseMask, // equal needs total ordering
                    spv::MemorySemanticsMaskNone, // unequal no memory ordering
                    end,
                    expected
                );
            } while (begin!=expected);
            accessor.set(0,begin);
            accessor.set(1,end);
        }
        // broadcast the span to everyone
        nbl::hlsl::glsl::barrier();
        bool took = false;
        if (active)
        {
            uint32_t begin;
            uint32_t end;
            accessor.get(0,begin);
            accessor.get(1,end);
            begin += endOffsetInPopped;
            if (begin<=end)
            {
                val = vk::RawBufferLoad<T>(getStorage(begin-1));
                took = true;
            }
        }
        return took;
    }
    template<typename BroadcastAccessor>
    bool pop(BroadcastAccessor accessor, const bool active, out T val, const uint16_t endOffsetInPopped, const bool electedInvocation)
    {
        // TOOD: replace `atomicIAdd(p,0)` with `atomicLoad(p)`
        const uint32_t beginHint = spirv::atomicIAdd((counters+PoppedOffset).deref().__get_spv_ptr(),spv::ScopeWorkgroup,spv::MemorySemanticsMaskNone,0u);
        return pop(accessor,active,val,endOffsetInPopped,electedInvocation,beginHint);
    }

    bda::__ptr<uint32_t> counters;
    uint64_t pStorage;
    uint16_t capacityLog2;
};

}
}
#endif