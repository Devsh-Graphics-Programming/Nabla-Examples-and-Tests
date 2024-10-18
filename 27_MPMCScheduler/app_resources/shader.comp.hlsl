//#include "nbl/builtin/hlsl/memory_accessor.hlsl"
//#include "nbl/builtin/hlsl/type_traits.hlsl"

//#include "schedulers/mpmc.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

#include "common.hlsl"

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"


using namespace nbl::hlsl;


#if 0
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
#endif

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
//NBL_REGISTER_OBJ_TYPE(WhittedTask,8);

#if 0
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
    
    void set(const uint32_t ix, const in uint32_t val)
    {
        sdata[ix] = val;
    }
    void get(const uint32_t ix, out uint32_t val)
    {
        val = sdata[ix];
    }
};
static nbl::hlsl::MPMCScheduler<WhittedTask,8*8,SharedAccessor> scheduler;

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
#endif

[[vk::binding(0,0)]] RWTexture2D<uint32_t> framebuffer;

void WhittedTask::operator()()
{
#if 0
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
//                scheduler.push(newTask);
            }
            // deal with refraction
            if (isGlass)
            {
                newThroughput -= throughput;
                newThroughput *= color;
                newTask.setThroughput(newThroughput);
                newTask.setRayDir(nbl_glsl_refract(-rayDir,normal,backside,NdotV,rcpOrientedEta));
//                scheduler.push(newTask);

            }
        }
        else
            contribution = throughput*color;
    }
    else // miss
        contribution = throughput*(rayDir.y>0.f ? float32_t3(0.1,0.7,0.03):float32_t3(0.05,0.25,1.0));

    if (contribution.r+contribution.g+contribution.b<1.f/2047.f)
        return;
#endif

    // Use device traits to do CAS loops on R32_UINT view of RGB9E5 when no VK_NV_shader_atomic_float16_vector
//    spirv::atomicAdd(spirv::addrof(framebuffer),contribution);
    framebuffer[uint32_t2(outputX,outputY)] = 0xffFFffFFu;
}

// move to `nbl/builtin/hlsl/shared_exp_t3.hlsl`

template<typename UintT, uint16_t ExponentBits>
struct shared_exp_t3
{
    using this_t = shared_exp_t3;

    UintT storage;
};

/*
uvec3 nbl_glsl_impl_sharedExponentEncodeCommon(in vec3 clamped, in int newExpBias, in int newMaxExp, in int mantissaBits, out int shared_exp)
{
    const float maxrgb = max(max(clamped.r, clamped.g), clamped.b);
    // TODO: optimize this
    const int f32_exp = int(nbl_glsl_ieee754_extract_biased_exponent(maxrgb)) - 126;

    shared_exp = clamp(f32_exp, -newExpBias, newMaxExp + 1);

    float scale = exp2(mantissaBits - shared_exp);
    const uint maxm = uint(maxrgb * scale + 0.5);
    const bool need = maxm == (0x1u << mantissaBits);
    scale = need ? 0.5 * scale : scale;
    shared_exp = need ? (shared_exp + 1) : shared_exp;
    return uvec3(clamped * scale + vec3(0.5));
}

uvec2 nbl_glsl_encodeRGB9E5(in vec3 col)
{
    const vec3 clamped = clamp(col, vec3(0.0), vec3(nbl_glsl_MAX_RGB19E7));

    int shared_exp;
    const uvec3 mantissas = nbl_glsl_impl_sharedExponentEncodeCommon(clamped, nbl_glsl_RGB19E7_EXP_BIAS, nbl_glsl_MAX_RGB19E7_EXP, nbl_glsl_RGB19E7_MANTISSA_BITS, shared_exp);

    uvec2 encoded;
    encoded.x = bitfieldInsert(mantissas.x, mantissas.y, nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[1], nbl_glsl_RGB19E7_G_COMPONENT_SPLIT);
    encoded.y = bitfieldInsert(
        mantissas.y >> nbl_glsl_RGB19E7_G_COMPONENT_SPLIT,
        mantissas.z,
        nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[2],
        nbl_glsl_RGB19E7_MANTISSA_BITS)
        | uint((shared_exp + nbl_glsl_RGB19E7_EXP_BIAS) << nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[3]);

    return encoded;
}
*/


struct Dummy
{
    void operator()()
    {
        next();
    }

    WhittedTask next;
    bool nextValid;
};
static Dummy scheduler;

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
        uint32_t2 GlobalInvocationID = glsl::gl_WorkGroupID().xy*glsl::gl_WorkGroupSize().xy;
        // TODO: morton code 
        {
            const uint32_t linearIx = glsl::gl_LocalInvocationIndex();
            GlobalInvocationID.x += linearIx%WorkgroupSizeX;
            GlobalInvocationID.y += linearIx/WorkgroupSizeX;
        }
#if 0
        scheduler.next.origin = float32_t3(0,0,-5);
        scheduler.next.setThroughput(float32_t3(1,1,1));
        scheduler.next.outputX = GlobalInvocationID.x;
        scheduler.next.outputY = GlobalInvocationID.y;
        {
            using namespace nbl::hlsl;
            float32_t3 ndc;
            {
                const float32_t2 totalInvocations = glsl::gl_NumWorkGroups().xy*8.f;
                ndc.xy = (float32_t2(GlobalInvocationID.xy)+float32_t2(0.5,0.5))*2.f/totalInvocations-float32_t2(1,1);
                ndc.y *= totalInvocations.y/totalInvocations.x; // aspect raio
            }
            ndc.z = 1.f; // FOV of 90 degrees
            scheduler.next.setRayDir(normalize(ndc));
        }
        scheduler.next.depth = 0;
#endif
//        scheduler.sharedAcceptableIdleCount = 0;
//        scheduler.globalAcceptableIdleCount = 0;
        scheduler.nextValid = true;
    }

    // excute implcit as scheduled
//    scheduler();
}
