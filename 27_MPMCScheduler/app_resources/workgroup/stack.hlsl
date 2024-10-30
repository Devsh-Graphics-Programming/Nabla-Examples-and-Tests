#ifndef _NBL_HLSL_WORKGROUP_STACK_HLSL_
#define _NBL_HLSL_WORKGROUP_STACK_HLSL_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{

// You are to separte the `push` from `pop` by a barrier!!!
template<typename T, typename SharedAccessor, uint32_t BaseOffset=0> // TODO: get rid of the base offset
struct Stack
{
    static const uint32_t SizeofTInDWORDs = (sizeof(T)-1)/sizeof(uint32_t)+1;

    struct Span
    {
        uint32_t begin;
        uint32_t end;
    };

    uint32_t capacity()
    {
        return (accessor.size()-StorageOffset)/SizeofTInDWORDs;
    }

    bool push(T el)
    {
        const uint32_t cursor = accessor.atomicAdd(CursorOffset,1);
        const uint32_t offset = StorageOffset+cursor*SizeofTInDWORDs;
        if (offset+SizeofTInDWORDs>accessor.size())
            return false;
        accessor.set(offset,el);
        return true;
    }

    // everything here is must be done by one invocation between two barriers, all invocations must call this method
    // `electedInvocation` must true only for one invocation and be such that `endOffsetInPopped` has the highest value amongst them
    bool pop(const bool active, out T val, const uint16_t endOffsetInPopped, const bool electedInvocation)
    {
        if (electedInvocation)
        {
            uint32_t cursor;
            accessor.get(CursorOffset,cursor);
            const uint32_t StackEnd = capacity();
            // handle past overflows (trim them)
            if (cursor>StackEnd)
                cursor = StackEnd;
            accessor.set(SpanEndOffset,cursor);
            // set the cursor to where we can push to in the future
            cursor = cursor>endOffsetInPopped ? (cursor-endOffsetInPopped):0;
            accessor.set(CursorOffset,cursor);
        }
        // broadcast the span to everyone
        glsl::barrier();
        uint32_t srcIx;
        bool took = false;
        if (active)
        {
            uint32_t srcIx,spanEnd;
            accessor.get(CursorOffset,srcIx);
            accessor.get(SpanEndOffset,spanEnd);
            srcIx += endOffsetInPopped;
            if (srcIx<=spanEnd)
            {
                accessor.get(srcIx-1,val);
                took = true;
            }
        }
        // make it safe to push again (no race condition on overwrites)
        glsl::barrier();
        return took;
    }

    static const uint32_t CursorOffset = BaseOffset+0;
    static const uint32_t SpanEndOffset = BaseOffset+1;
    static const uint32_t StorageOffset = BaseOffset+2;
    SharedAccessor accessor;
};

}
}
}
#endif