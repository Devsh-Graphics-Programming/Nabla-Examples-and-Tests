#ifndef _NBL_HLSL_WORKGROUP_POOL_ALLOCATOR_HLSL_
#define _NBL_HLSL_WORKGROUP_POOL_ALLOCATOR_HLSL_

#include "workgroup/stack.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{

// You are to separte the `push` from `pop` by a barrier!!!
template<typename T, typename SharedAccessor, uint32_t BaseOffset=0>
struct PoolAllocator
{
    // TODO: need better accessors for `uint16_t` addresses (atomicAnd and atomicOr)
    using value_type = uint32_t;

    void init()
    {
        const uint32_t3 WorkGroupSize = glsl::gl_WorkGroupSize();
        const uint32_t InvocationCount = WorkGroupSize.x * WorkGroupSize.y * WorkGroupSize.z;
        const uint32_t newSize = freeAddressStack.capacity();
        glsl::barrier();
        // prime the stack with free addresses
        for (uint32_t virtIx=glsl::gl_LocalInvocationIndex(); virtIx<newSize; virtIx+=InvocationCount)
            freeAddressStack.accessor.set(freeAddressStack.StorageOffset,virtIx);
        if (glsl::gl_LocalInvocationIndex()==0)
            freeAddressStack.accessor.set(freeAddressStack.CursorOffset,newSize);
        glsl::barrier();
    }

    // needs to be called cooperatively, see the documentation of `workgroup::Stack`
    bool alloc(const bool active, out value_type addr, const uint16_t endOffsetInPopped, const bool electedInvocation)
    {
        return freeAddressStack.pop(addr);
    }

    bool free(value_type addr) {freeAddressStack.push(addr);}

    Stack<uint32_t,SharedAccessor,BaseOffset> freeAddressStack;
};

}
}
}
#endif