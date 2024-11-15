#ifndef _NBL_HLSL_SCHEDULERS_MPMC_HLSL_
#define _NBL_HLSL_SCHEDULERS_MPMC_HLSL_

//#include "../workgroup/stack.hlsl"
//#include "mpmc_queue.hlsl"

#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"

namespace nbl
{
namespace hlsl
{
namespace schedulers
{

// TODO: improve and use a Global Pool Allocator and stop moving whole payloads around in VRAM
template<typename Task, uint32_t WorkGroupSize, typename SharedAccessor, class device_capabilities=void>
struct MPMC
{
    // TODO: static asset that the signature of the `Task::operator()` is `void()`
    static const uint16_t BallotDWORDS = workgroup::scratch_size_ballot<WorkGroupSize>::value;
    static const uint16_t ScanDWORDS = workgroup::scratch_size_arithmetic<WorkGroupSize>::value;
    static const uint16_t PopCountOffset = BallotDWORDS+ScanDWORDS;

    void push(const in Task payload)
    {
        // already stole some work, need to spill
        if (nextValid)
        {
#if 0
            // if the shared memory stack will overflow
            if (!sStack.push(payload))
            {
                // spill to a global queue
                gQueue.push(payload);
            }
#endif
        }
        else
            next = payload;
    }

#if 0
    // returns if there's any invocation at all that wants to pop
    uint16_t popCountInclusive_impl(out uint16_t reduction)
    {
        // ballot how many items want to be taken
        workgroup::ballot(!nextValid,sStack.accessor);
        // clear the count
        sStack.accessor.set(PopCountOffset,0);
        glsl::barrier();
        // prefix sum over the active requests
        using ArithmeticAccessor = accessor_adaptors::Offset<SharedAccessor,uint32_t,integral_constant<uint32_t,BallotDWORDS> >;
        ArithmeticAccessor arithmeticAccessor;
        arithmeticAccessor.accessor = sStack.accessor;
        const uint16_t retval = workgroup::ballotInclusiveBitCount<WorkGroupSize,SharedAccessor,ArithmeticAccessor,device_capabilities>(sStack.accessor,arithmeticAccessor);
        sStack.accessor = arithmeticAccessor.accessor;
        // get the reduction
        if (glsl::gl_LocalInvocationIndex()==(WorkGroupSize-1))
            sStack.accessor.set(PopCountOffset,retval);
        glsl::barrier();
        sStack.accessor.get(PopCountOffset,reduction);
        return retval;
    }
#endif

    void operator()()
    {
        const bool lastInvocationInGroup = glsl::gl_LocalInvocationIndex()==(WorkGroupSize-1);
        // need to quit when we don't get any work, otherwise we'd spin expecting forward progress guarantees
        for (uint16_t popCount=0xffffu; popCount; )
        {
            if (nextValid) // this invocation has some work to do
            {
                // ensure by-value semantics, the task may push work itself
                Task tmp = next;
                nextValid = false;
                tmp();
            }
#if 0
            // everyone sync up here so we can count how many invocations won't have jobs
            glsl::barrier();
            uint16_t popCountInclusive = popCountInclusive_impl(popCount);
            // now try and pop work from out shared memory stack
            if (popCount > sharedAcceptableIdleCount)
            {
                // look at the way the `||` is expressed, its specifically that way to avoid short circuiting!
                nextValid = sStack.pop(!nextValid,next,popCountInclusive,lastInvocationInGroup) || nextValid;
                // now if there's still a problem, grab some tasks from the global ring-buffer
                popCountInclusive = popCountInclusive_impl(popCount);
                if (popCount > globalAcceptableIdleCount)
                {
                    // reuse the ballot smem for broadcasts, nobody need the ballot state now
                    gQueue.pop(sStack.accessor,!nextValid,next,popCountInclusive,lastInvocationInGroup,0);
                }
            }
#else
            popCount = 0;
#endif
        }
    }

//    MPMCQueue<Task> gQueue;
//    workgroup::Stack<Task,SharedAccessor,PopCountOffset+1> sStack;
    Task next;
    // popping work from the stack and queue might be expensive, expensive enough to not justify doing all the legwork to just pull a few items of work
    uint16_t sharedAcceptableIdleCount;
    uint16_t globalAcceptableIdleCount;
    bool nextValid;
};

}
}
}
#endif