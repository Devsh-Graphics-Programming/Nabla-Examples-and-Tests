//#include "nbl/builtin/hlsl/memory_accessor.hlsl"

#include "common.hlsl"

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"


using namespace nbl::hlsl;

// Scene
enum Material : uint32_t
{
    Emission = 0,
    Metal,
    Glass
};
struct Sphere
{
    static const uint32_t MaxColorValue = 1023;

    float16_t3 getColor()
    {
        return float16_t3(R,G,B)/float16_t3(MaxColorValue,MaxColorValue,MaxColorValue);
    }

    float32_t intersect(const float32_t3 rayOrigin, const float32_t3 rayDir)
    {
        float32_t3 relOrigin = rayOrigin - position;
        float32_t relOriginLen2 = dot(relOrigin,relOrigin);

        float32_t dirDotRelOrigin = dot(rayDir,relOrigin);
        float32_t det = radius2 - relOriginLen2 + dirDotRelOrigin * dirDotRelOrigin;

        // do some speculative math here
        float32_t detsqrt = sqrt(det);
        return -dirDotRelOrigin + (relOriginLen2 > radius2 ? (-detsqrt) : detsqrt);
    }

    float32_t3 position;
    float32_t radius2;
    uint32_t R : 10;
    uint32_t G : 10;
    uint32_t B : 10;
    uint32_t material : 2;
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
        float32_t3(-1,1,0),
        0.6f,
        Sphere::MaxColorValue,
        Sphere::MaxColorValue,
        0,
        Material::Metal
    },
    {
        float32_t3(1,1,0),
        0.8f,
        0,
        Sphere::MaxColorValue,
        Sphere::MaxColorValue,
        Material::Metal
    },
    // Glass balls need to be monochromatic, cause I didn't do RGB in my task payload
    {
        float32_t3(-2,3,0),
        0.7f,
        Sphere::MaxColorValue,
        Sphere::MaxColorValue,
        Sphere::MaxColorValue,
        Material::Glass
    },
    {
        float32_t3(2,3,0),
        0.7f,
        0,
        Sphere::MaxColorValue/2,
        0,
        Material::Glass
    }
};

#include "nbl/builtin/hlsl/format/octahedral.hlsl"

// Payload and Executor for our Task-Graph
struct WhittedTask
{
    static const uint32_t MaxDepth = (1<<5)-1;

    using dir_t = format::octahedral<uint32_t>;

    float32_t3 origin;
    dir_t dir;
    float16_t3 throughput; 
    float16_t3 contribution;
    //
    uint32_t outputX : 14;
    uint32_t outputY : 13;
    uint32_t depth : 5;

    void setRayDir(float32_t3 _dir)
    {
        dir = _static_cast<dir_t>(_dir);
    }
    float32_t3 getRayDir() 
    {
        return _static_cast<float32_t3>(dir);
    }

    // yay workaround for https://github.com/microsoft/DirectXShaderCompiler/issues/6973
    void __impl_call();

    void operator()() {__impl_call();}
};
//NBL_REGISTER_OBJ_TYPE(WhittedTask,8);

// something something, Nvidia can do 32 bytes of smem per invocation
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
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
    
    void set(const uint32_t ix, const in uint32_t val)
    {
        sdata[ix] = val;
    }
    void get(const uint32_t ix, out uint32_t val)
    {
        val = sdata[ix];
    }
};

//
#include "schedulers/mpmc.hlsl"
struct SubgroupCaps
{
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = true;
};
static nbl::hlsl::schedulers::MPMC<WhittedTask,WorkgroupSizeX*WorkgroupSizeY,SharedAccessor,SubgroupCaps> scheduler;

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



#include "nbl/builtin/hlsl/format/shared_exp.hlsl"
[[vk::binding(0,0)]] RWTexture2D<uint32_t> framebuffer;

void WhittedTask::__impl_call()
{
    const float32_t3 rayDir = getRayDir();

    // intersect with spheres
    uint32_t closestIx = SphereCount;
    const float NoHit = bit_cast<float32_t>(numeric_limits<float32_t>::infinity);
    float closestD = NoHit;
    if (depth<MaxDepth)
    for (uint32_t i=0; i<SphereCount; i++)
    {
        const Sphere sphere = spheres[i];
        const float32_t d = sphere.intersect(origin,rayDir);
        if (d>0 && d<closestD)
        {
            closestIx = i;
            closestD = d;
        }
    }

    if (closestD<NoHit)
    {
        const Sphere sphere = spheres[closestIx];
        const float16_t3 color = sphere.getColor();
        if (sphere.material!=Material::Emission)
        {
            const float32_t3 hitPoint = origin+rayDir*closestD;
            const float32_t3 normal = (hitPoint-sphere.position)*rsqrt(sphere.radius2);
            const float32_t NdotV = dot(-rayDir,normal);
            float orientedEta, rcpOrientedEta;
            const bool backside = nbl_glsl_getOrientedEtas(orientedEta,rcpOrientedEta,NdotV,1.333f);

            const bool isGlass = sphere.material==Material::Glass;
            WhittedTask newTask = this;
            newTask.depth++;
            newTask.origin = hitPoint+normal*0.0001;

            // deal with reflection
            float16_t3 newThroughput = throughput;
            // fresnel
            float32_t fresnel;
            if (isGlass)
            {
                const float32_t F0 = 0.08f;
                float32_t fresnel = nbl_glsl_fresnel_dielectric_common(orientedEta*orientedEta,abs(NdotV));
                newThroughput *= float16_t(fresnel);
            }
            // push reflection ray
            {
                const float32_t3 reflected = 2.f*normal+rayDir;

                newTask.throughput = newThroughput;
                if (!isGlass)
                    newTask.throughput *= color;
                newTask.setRayDir(reflected);
                scheduler.push(newTask);
            }
            // deal with refraction
            if (isGlass)
            {
                newTask.throughput -= throughput;
                newTask.throughput *= color;
                newTask.setRayDir(nbl_glsl_refract(-rayDir,normal,backside,NdotV,rcpOrientedEta));
//                scheduler.push(newTask);

            }
            // we'll keep counting up the contribution
            return;
        }
        else
            contribution += throughput*color;
    }
    else // miss
        contribution += throughput*(rayDir.y<0.f ? float16_t3(0.1,0.7,0.03):float16_t3(0.05,0.25,1.0));

    if (contribution.r+contribution.g+contribution.b<1.f/2047.f)
        return;

    const uint32_t2 output = uint32_t2(outputX,outputY);
    // CAS loops on R32_UINT view of RGB9E5 because there's no atomic add (only NV has vector half float atomics, but RGB9E5 not storable)
    using rgb9e5_t = format::shared_exp<uint32_t,3,5>;
    rgb9e5_t actual,expected;
    // assume image is empty, good assumption always true once
    actual.storage = 0;
    do
    {
        expected = actual;
        rgb9e5_t newVal = _static_cast<rgb9e5_t>(_static_cast<float32_t3>(expected)+float32_t3(contribution));
        InterlockedCompareExchange(framebuffer[output],expected.storage,newVal.storage,actual.storage);
    } while (expected!=actual);
}

[[vk::push_constant]] PushConstants pc;

// have to do weird stuff with workgroup size because of subgroup full spec
namespace nbl
{
namespace hlsl
{
namespace glsl
{
uint32_t3 gl_WorkGroupSize() {return uint32_t3(WorkgroupSizeX*WorkgroupSizeY,1,1);}
}
}
}
[numthreads(WorkgroupSizeX*WorkgroupSizeY,1,1)]
void main()
{
    // manually push an explicit workload
    {
        // reconstruct the actual XY coordinate we want
        const uint32_t2 VirtualWorkgroupSize = uint32_t2(WorkgroupSizeX,WorkgroupSizeY);
        uint32_t2 GlobalInvocationID = glsl::gl_WorkGroupID().xy*VirtualWorkgroupSize;
        // TODO: morton code 
        {
            const uint32_t linearIx = glsl::gl_LocalInvocationIndex();
            GlobalInvocationID.x += linearIx%WorkgroupSizeX;
            GlobalInvocationID.y += linearIx/WorkgroupSizeX;
        }
        scheduler.next.origin = float32_t3(0,2.5,6);
        scheduler.next.throughput = float16_t3(1,1,1);
        scheduler.next.contribution = float16_t3(0,0,0);
        scheduler.next.outputX = GlobalInvocationID.x;
        scheduler.next.outputY = GlobalInvocationID.y;
        {
            using namespace nbl::hlsl;
            float32_t3 ndc;
            {
                const float32_t2 totalInvocations = glsl::gl_NumWorkGroups().xy*VirtualWorkgroupSize;
                ndc.xy = float32_t2(GlobalInvocationID.xy)*float32_t2(2,-2)/totalInvocations+float32_t2(-1.f,1.f)+float32_t2(1,1)/totalInvocations;
                ndc.y *= totalInvocations.y/totalInvocations.x; // aspect raio
            }
            ndc.z = -1.f; // FOV of 90 degrees
            scheduler.next.setRayDir(ndc);
        }
        scheduler.next.depth = 0;
        scheduler.sharedAcceptableIdleCount = 0;
        scheduler.globalAcceptableIdleCount = 0;
        scheduler.nextValid = true;
    }

    // excute implcit as scheduled
    scheduler();
}
