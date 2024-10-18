#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"


namespace nbl
{
namespace hlsl
{

// move: `workgroup/mpmc_queue`
namespace workgroup
{

// You are to separte the `push` from `pop` by a barrier!!!
template<typename T, typename SharedAccessor, uint32_t BaseOffset=0>
struct Stack
{
    static const uint32_t SizeofTInDWORDs = (sizeof(T)-1)/sizeof(uint32_t)+1;

    struct Span
    {
        uint32_t begin;
        uint32_t end;
    };

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
            const uint32_t StackEnd = (accessor.size()-StorageOffset)/SizeofTInDWORDs;
            // handle past overflows (trim them)
            if (cursor>StackEnd)
                cursor = StackEnd;
            accessor.set(SpanEndOffset,cursor);
            // set the cursor to where we can push to in the future
            cursor = cursor>endOffsetInPopped ? (cursor-endOffsetInPopped):0;
            accessor.set(CursorOffset,cursor);
        }
        // broadcast the span to everyone
        nbl::hlsl::glsl::barrier();
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
        nbl::hlsl::glsl::barrier();
        return took;
    }

    static const uint32_t CursorOffset = BaseOffset+0;
    static const uint32_t SpanEndOffset = BaseOffset+1;
    static const uint32_t StorageOffset = BaseOffset+2;
    SharedAccessor accessor;
};
}

// move: `mpmc_queue`
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
        spirv::atomicIAdd(committed.__get_spv_ptr(),spv::ScopeWorkgroup,spv::MemorySemanticsReleaseMask,1);
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

// move: `scheduler/mpmc`
template<typename Task, uint32_t WorkGroupSize, typename SharedAccessor, typename GlobalQueue, class device_capabilities=void>
struct MPMCScheduler
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
            // if the shared memory stack will overflow
            if (!sStack.push(payload))
            {
                // spill to a global queue
                gQueue.push(payload);
            }
        }
        else
            next = payload;
    }

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
        sStack.accessor.set(PopCountOffset,retval);
        glsl::barrier();
        sStack.accessor.get(PopCountOffset,retval);
        return retval;
    }

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
            // everyone sync up here so we can count how many invocations won't have jobs
            glsl::barrier();
            uint16_t popCountInclusive = popCountInclusive_impl(popCount);
            // now try and pop work from out shared memory stack
            if (popCount)
            {
                // look at the way the `||` is expressed, its specifically that way to avoid short circuiting!
                nextValid = sStack.pop(!nextValid,next,popCountInclusive,lastInvocationInGroup) || nextValid;
                // now if there's still a problem, grab some tasks from the global ring-buffer
                popCountInclusive = popCountInclusive_impl(popCount);
                if (popCount)
                {
                    // reuse the ballot smem for broadcasts, nobody need the ballot state now
                    gQueue.pop(sStack.accessor,!nextValid,next,popCountInclusive,lastInvocationInGroup,0);
                }
            }
        }
    }

    MPMCQueue<Task> gQueue;
    workgroup::Stack<Task,SharedAccessor,PopCountOffset+1> sStack;
    Task next;
    bool nextValid;
};

}
}


// ================================ SHADER START ================================
[[vk::binding(0,0)]] RWTexture2D<float32_t4> framebuffer;

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"

enum Material : uint32_t
{
    Emission = 0,
    Metal,
    Glass
};
struct Sphere
{
    static const uint32_t MaxColorValue = 1023;

    float32_t3 getColor()
    {
        return float32_t3(R,G,B)/float32_t3(MaxColorValue,MaxColorValue,MaxColorValue);
    }

    float32_t3 position;
    float32_t radius;
    uint32_t R : 10;
    uint32_t G : 10;
    uint32_t B : 10;
    Material material : 2;
};
const static uint32_t SphereCount = 5;
const static Sphere spheres[5] = {
    {
        float32_t3(0,5,0),
        0.5f,
        Sphere::MaxColorValue,
        0,
        0,
        Material::Emission
    },
    {
        float32_t3(-2,3,0),
        0.6f,
        Sphere::MaxColorValue,
        Sphere::MaxColorValue,
        0,
        Material::Metal
    },
    {
        float32_t3(2,3,0),
        0.4f,
        0,
        Sphere::MaxColorValue,
        Sphere::MaxColorValue,
        Material::Metal
    },
    // Glass balls need to be monochromatic, cause I didn't do RGB in my task payload
    {
        float32_t3(-1,1,0),
        0.7f,
        Sphere::MaxColorValue,
        Sphere::MaxColorValue,
        Sphere::MaxColorValue,
        Material::Glass
    },
    {
        float32_t3(-1,1,0),
        0.7f,
        0,
        Sphere::MaxColorValue/2,
        0,
        Material::Glass
    }
};

struct WhittedTask
{
    static const uint32_t MaxDepth = (1<<2)-1;
    static const uint32_t MaxTheta = (1<<19)-1;
    static const uint32_t MaxPhi = (1<<20)-1;

    float32_t3 origin;
    uint32_t throughputR : 11;
    uint32_t throughputG : 11;
    uint32_t throughputB : 10;
    //
    uint64_t outputX : 12;
    uint64_t outputY : 11;
    uint64_t dirTheta : 19;
    uint64_t dirPhi : 20;
    uint64_t depth : 2;

    void setThroughput(const float32_t3 col)
    {
        throughputR = uint32_t(col.r*2047.f+0.4f);
        throughputG = uint32_t(col.g*2047.f+0.4f);
        throughputB = uint32_t(col.b*1023.f+0.4f);
    }
    float32_t3 getThroughput()
    {
        return float32_t3(throughputR,throughputG,throughputB)/float32_t3(2047,2047,1023);
    }

    void setRayDir(float32_t3 dir)
    {
        const float32_t pi = nbl::hlsl::numbers::pi<float32_t>;
        dirTheta = acos(dir.z)*float32_t(MaxTheta)/pi+0.5f;
        // rely on integer wraparound to map (-pi,0) to [UINT_MAX,INT_MAX)
        dirPhi = uint32_t(floor(atan2(dir.x,dir.y)*float32_t(MaxPhi)/pi+0.5f));
    }
    float32_t3 getRayDir()
    {
        float32_t3 dir;
        const float32_t pi = nbl::hlsl::numbers::pi<float32_t>;
        dir.z = cos(float32_t(dirTheta)*pi/float32_t(MaxTheta));
        // shtuff
        {
            const float32_t phi = float32_t(dirPhi)/float32_t(MaxPhi);
            dir.xy = float32_t2(cos(phi),sin(phi));
        }
        return dir;
    }

    void operator()();
};
NBL_REGISTER_OBJ_TYPE(WhittedTask,8);

struct GlobalAccessor
{
};
// something something, Nvidia can do 32 bytes of smem per invocation
groupshared uint32_t sdata[512];
struct SharedAccessor
{
    static uint32_t size() {return 512;}

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }

    uint32_t atomicOr(const uint32_t ix, const uint32_t val)
    {
        return nbl::hlsl::glsl::atomicOr(sdata[ix],val);
    }
    uint32_t atomicAdd(const uint32_t ix, const uint32_t val)
    {
        return nbl::hlsl::glsl::atomicAdd(sdata[ix],val);
    }
    
    template<typename T>
    void set(const uint32_t ix, const in T val)
    {
//        sdata[ix] = val;
    }
    template<typename T>
    void get(const uint32_t ix, out T val)
    {
//        sdata[ix] = val;
    }
};
static nbl::hlsl::MPMCScheduler<WhittedTask,8*8,SharedAccessor,GlobalAccessor> scheduler;

// stolen from Nabla GLSL
bool nbl_glsl_getOrientedEtas(out float orientedEta, out float rcpOrientedEta, in float NdotI, in float eta)
{
    const bool backside = NdotI<0.0;
    const float rcpEta = 1.0/eta;
    orientedEta = backside ? rcpEta:eta;
    rcpOrientedEta = backside ? eta:rcpEta;
    return backside;
}
float nbl_glsl_fresnel_dielectric_common(in float orientedEta2, in float AbsCosTheta)
{
    const float SinTheta2 = 1.0-AbsCosTheta*AbsCosTheta;

    // the max() clamping can handle TIR when orientedEta2<1.0
    const float t0 = sqrt(max(orientedEta2-SinTheta2,0.0));
    const float rs = (AbsCosTheta - t0) / (AbsCosTheta + t0);

    const float t2 = orientedEta2 * AbsCosTheta;
    const float rp = (t0 - t2) / (t0 + t2);

    return (rs * rs + rp * rp) * 0.5;
}
float32_t3 nbl_glsl_refract(in float32_t3 I, in float32_t3 N, in bool backside, in float NdotI, in float rcpOrientedEta)
{
    const float NdotI2 = NdotI*NdotI;
    const float rcpOrientedEta2 = rcpOrientedEta*rcpOrientedEta;
    const float abs_NdotT = sqrt(rcpOrientedEta2*NdotI2 + 1.0 - rcpOrientedEta2);
    const float NdotT = backside ? abs_NdotT:(-abs_NdotT);
    return N*(NdotI*rcpOrientedEta + NdotT) - rcpOrientedEta*I;
}

void WhittedTask::operator()()
{
    using namespace nbl::hlsl;

    const float32_t3 rayDir = getRayDir();
    const float32_t3 throughput = getThroughput();

    // intersect with spheres
    uint32_t closestIx = SphereCount;
    const float NoHit = bit_cast<float32_t>(numeric_limits<float32_t>::infinity);
    float closestD = NoHit;
    if (depth<MaxDepth)
    for (uint32_t i=0; i<SphereCount; i++)
    {
        const Sphere sphere = spheres[i];
// TODO intersection
    }

    float32_t3 contribution = float32_t3(0,0,0);
    if (closestD<NoHit)
    {
        const Sphere sphere = spheres[closestIx];
        const float32_t3 color = sphere.getColor();
        if (sphere.material!=Material::Emission)
        {
            const float32_t3 hitPoint = origin+rayDir*closestD;
            const float32_t3 normal = (hitPoint-sphere.position)/sphere.radius;
            const float32_t NdotV = dot(-rayDir,normal);
            float orientedEta, rcpOrientedEta;
            const bool backside = nbl_glsl_getOrientedEtas(orientedEta,rcpOrientedEta,NdotV,1.333f);

            const bool isGlass = sphere.material==Material::Glass;
            WhittedTask newTask = this;
            newTask.depth++;
            newTask.origin = hitPoint;

            // deal with reflection
            float32_t3 newThroughput = throughput;
            // fresnel
            float32_t fresnel;
            if (isGlass)
            {
                const float32_t F0 = 0.08f;
                float32_t fresnel = nbl_glsl_fresnel_dielectric_common(orientedEta*orientedEta,abs(NdotV));
                newThroughput *= fresnel;
            }
            // push reflection ray
            {
                const float32_t3 reflected = 2.f*normal+rayDir;

                newTask.setThroughput(isGlass ? newThroughput:(color*newThroughput));
                newTask.setRayDir(reflected);
                scheduler.push(newTask);
            }
            // deal with refraction
            if (isGlass)
            {
                newThroughput -= throughput;
                newThroughput *= color;
                newTask.setThroughput(newThroughput);
                newTask.setRayDir(nbl_glsl_refract(-rayDir,normal,backside,NdotV,rcpOrientedEta));
                scheduler.push(newTask);

            }
        }
        else
            contribution = throughput*color;
    }
    else // miss
        contribution = throughput*(rayDir.y>0.f ? float32_t3(0.1,0.7,0.03):float32_t3(0.05,0.25,1.0));

    if (contribution.r+contribution.g+contribution.b<1.f/2047.f)
        return;

    // Use device traits to do CAS loops on R32_UINT view of RGB9E5 when no VK_NV_shader_atomic_float16_vector
//    spirv::atomicAdd(spirv::addrof(framebuffer),contribution);
    framebuffer[uint32_t2(outputX,outputY)] = float32_t4(contribution,1.f);
}

namespace nbl
{
namespace hlsl
{
namespace glsl
{
uint32_t3 gl_WorkGroupSize() {return uint32_t3(8,8,1);}
}
}
}
[numthreads(8,8,1)]
void main(uint32_t3 gl_GlobalInvocationID : SV_DispatchThreadID)
{
    // manually push an explicit workload
    {
        scheduler.next.origin = float32_t3(0,0,-5);
        scheduler.next.setThroughput(float32_t3(1,1,1));
        scheduler.next.outputX = gl_GlobalInvocationID.x;
        scheduler.next.outputY = gl_GlobalInvocationID.y;
        {
            using namespace nbl::hlsl;
            float32_t3 ndc;
            {
                const float32_t2 totalInvocations = glsl::gl_NumWorkGroups().xy*8.f;
                ndc.xy = (float32_t2(gl_GlobalInvocationID.xy)+float32_t2(0.5,0.5))*2.f/totalInvocations-float32_t2(1,1);
                ndc.y *= totalInvocations.y/totalInvocations.x; // aspect raio
            }
            ndc.z = 1.f; // FOV of 90 degrees
            scheduler.next.setRayDir(normalize(ndc));
        }
        scheduler.next.depth = 0;
        scheduler.nextValid = true;
    }

    // excute implcit as scheduled
    scheduler();
#ifdef DEBUG
    printf("Workgroup Quit");
#endif
}
