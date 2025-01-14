#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

using namespace nbl::hlsl;




//! Everything below this line is userspace
#include "common.hlsl"

uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(WORKGROUP_SIZE, 1, 1); }

[[vk::binding(0)]]
Texture2D<float32_t4> input;
[[vk::binding(1)]]
RWTexture2D<float32_t4> output;

[[vk::push_constant]] PushConstants pc;


template<int32_t Chnls>
struct TextureProxy
{
    NBL_CONSTEXPR int32_t Channels = Chnls;
    using texel_t = vector<float32_t,Channels>;

    // divisions by PoT constant will optimize out nicely
    template<typename T>
    T get(const uint16_t channel, const uint16_t uv)
    {
        return spill[uv/WORKGROUP_SIZE][channel];
    }
    template<typename T>
	void set(const uint16_t channel, const uint16_t uv, T value)
	{
        spill[uv/WORKGROUP_SIZE][channel] = value;
	}

    void load()
    {
        const uint16_t end = linearSize();
        uint16_t ix = workgroup::SubgroupContiguousIndex();
        // because workgroups do scans cooperatively all spill values need sane defaults
        for (uint16_t i=0; i<SpillSize; ix+=WORKGROUP_SIZE)
            spill[i++] = ix<end ? ((texel_t)(input[position(ix)])):promote<texel_t>(0.f);
    }

    void store()
    {
        const uint16_t end = linearSize();
        uint16_t i = 0;
        // making sure that we don't store out of range
        for (uint16_t ix=workgroup::SubgroupContiguousIndex(); ix<end; ix+=WORKGROUP_SIZE)
        {
            float32_t4 tmp = float32_t4(0,0,0,1);
            for (int32_t ch=0; ch<Channels; ch++)
                tmp[ch] = spill[i][ch];
            i++;
            // TODO: inverse SRGB on `tmp` because we're writing to SRGB via UNORM view!
            output[position(ix)] = tmp;
        }
    }

    uint16_t linearSize()
    {
        uint32_t3 dims;
        input.GetDimensions(0, dims.x, dims.y, dims.z);
        return _static_cast<uint16_t>(dims[activeAxis]);
    }

    uint16_t2 position(uint16_t ix)
    {
        uint16_t2 pos;
        // use the compile time constant - although `glsl::gl_WorkGroupSize().x` would have probably optimized out too (for now)
        pos[activeAxis] = ix;
        pos[activeAxis^0x1] = _static_cast<uint16_t>(glsl::gl_WorkGroupID().x);
        return pos;
    }

    // whether we pas along X or Y
    uint16_t activeAxis;

    // round up, of course
    NBL_CONSTEXPR uint16_t SpillSize = (MAX_SCANLINE-1)/WORKGROUP_SIZE+1;
    texel_t spill[SpillSize];
};

// we always use `uint32_t`
groupshared uint32_t smem[MAX_SCANLINE];
// will always be bigger
//static_assert(workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value <= MAX_SCANLINE);

struct SharedMemoryProxy
{
    // these get used by BoxBlur
    template<typename T, typename I=uint16_t>
	enable_if_t<sizeof(T)==sizeof(uint32_t),T> get(const I idx)
	{
		return bit_cast<T>(smem[idx]);
	}
	template<typename T, typename I=uint16_t>
	enable_if_t<sizeof(T)==sizeof(uint32_t),void> set(const I idx, T value)
	{
        smem[idx] = bit_cast<uint32_t>(value);
	}

    // and these get used by Prefix Sum

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }
};

template<typename DataAccessor, typename SharedAccessor, uint16_t WorkgroupSize> // TODO: define concepts for the BoxBlur and apply constraints
struct BoxBlur
{
    void operator()(NBL_REF_ARG(DataAccessor) data, NBL_REF_ARG(SharedAccessor) scratch, const uint16_t channel)
    {
        const uint16_t end = data.linearSize();
        const uint16_t localInvocationIndex = workgroup::SubgroupContiguousIndex();

        // prefix sum
        // note the dynamically uniform loop condition 
        for (uint16_t baseIx=0; baseIx<end; )
        {
            const uint16_t ix = localInvocationIndex+baseIx;
            float32_t input = data.template get<float32_t>(channel,ix);
            // dynamically uniform condition
            if (baseIx!=0)
            {
                // take result of previous prefix sum and add it to first element here
                if (localInvocationIndex==0)
                    input += scratch.template get<float32_t>(baseIx-1);
            }
            const float32_t sum = input;//workgroup::inclusive_scan<plus<float32_t>,WorkgroupSize>::template __call<SharedAccessor>(input,shared);
            baseIx += WorkgroupSize;
            // if doing the last prefix sum, we need to barrier to stop aliasing of temporary scratch for `inclusive_scan` and our scanline
            if (baseIx>=end)
                scratch.workgroupExecutionAndMemoryBarrier();
            // save prefix sum results
            scratch.template set<float32_t>(ix,sum);
            // previous prefix sum must have finished before we ask for results
            scratch.workgroupExecutionAndMemoryBarrier();
        }

        const float32_t last = end-1;
        for (uint16_t ix=localInvocationIndex; ix<end; ix+=WorkgroupSize)
        {
            const float u = ix;
            // Exercercise for reader do the start-end taps with bilinear interpolation
            const float right = scratch.template get<float32_t,int32_t>(clamp(u+radius,0.f,last));
            const float left = scratch.template get<float32_t,int32_t>(clamp(u-radius,0.f,last));
            data.template set<float32_t>(channel,ix,right-left);
        }
    }

    vector<float32_t,DataAccessor::Channels> borderColor;
    float32_t radius;
    uint16_t wrapMode;
};

/*
void boxBlur(
    NBL_REF_ARG(TextureAccessor) texAccessor,
    NBL_REF_ARG(SharedAccessor) smemAccessor,
    uint16_t itemsPerThread,
    uint16_t width,
    bool flip // <-------------------------------------- YOUR BOX BLUR SHOULD BE AGNOSTIC TO THIS!
) {
    const uint16_t lastIdx = width - 1;
    const float32_t scale = 1.0 / (2.0 * radius + 1.0);
    const uint16_t radiusFl = (uint16_t)floor(radius);
    const uint16_t radiusCl = (uint16_t)ceil(radius);
    const float32_t alpha = radius - radiusFl;


    for (uint16_t i = 0; i < itemsPerThread; i++)
    {
        uint16_t scanIdx = (i * glsl::gl_WorkGroupSize().x) + workgroup::SubgroupContiguousIndex();
        if (scanIdx > lastIdx) break;

        const float32_t leftIdx = scanIdx - radius;
        const float32_t rightIdx = scanIdx + radius;

        float32_t result = 0;
        if (rightIdx <= lastIdx)
        {
            float32_t rightFloor, rightCeil;
            smemAccessor.get(scanIdx + radiusFl, rightFloor);
            smemAccessor.get(scanIdx + radiusCl, rightCeil);
            result += lerp(rightFloor, rightCeil, alpha);
        }
        else switch (wrapMode)
        {
            case WRAP_MODE_CLAMP_TO_BORDER:
            {
                result += (rightIdx - lastIdx) * borderColor[channel];
            } break;
            case WRAP_MODE_CLAMP_TO_EDGE:
            {
                float32_t last, lastMinus1;
                smemAccessor.get(lastIdx, last);
                smemAccessor.get(lastIdx - 1, lastMinus1);
                result += (rightIdx - lastIdx) * (last - lastMinus1);
            } break;
            case WRAP_MODE_REPEAT:
            {
                const uint16_t repeatedIdx = (uint16_t)(rightIdx % width);
                float32_t repeatedValue;
                smemAccessor.get(repeatedIdx, repeatedValue);
                result += repeatedValue;
            } break;
            case WRAP_MODE_MIRROR:
            {
                const uint16_t mirroredIdx = (uint16_t)((rightIdx / width) % 2 == 0 ? rightIdx % width : lastIdx - (rightIdx % width));
                float32_t mirrored;
                smemAccessor.get(mirroredIdx, mirrored);
                result += mirrored;
            } break;
        }

        if (leftIdx >= 0)
        {
            float32_t leftFloor, leftCeil;
            smemAccessor.get(scanIdx - radiusFl, leftFloor);
            smemAccessor.get(scanIdx - radiusCl, leftCeil);
            result -= lerp(leftFloor, leftCeil, alpha);
        }
        else switch (wrapMode)
        {
            case WRAP_MODE_CLAMP_TO_BORDER:
            {
                result -= leftIdx * borderColor[channel];
            } break;
            case WRAP_MODE_CLAMP_TO_EDGE:
            {
                float32_t first;
                smemAccessor.get(0, first);
                result -= abs(leftIdx) * first;
            } break;
            case WRAP_MODE_REPEAT:
            {
                const uint16_t repeatedIdx = (uint16_t)((leftIdx % width + width) % width);
                float32_t repeatedValue;
                smemAccessor.get(repeatedIdx, repeatedValue);
                result -= repeatedValue;
            } break;
            case WRAP_MODE_MIRROR:
            {
                const uint16_t mirroredIdx = (uint16_t)((-leftIdx / width) % 2 == 0 ? (-leftIdx) % width : lastIdx - ((-leftIdx) % width));
                float32_t mirrored;
                smemAccessor.get(mirroredIdx, mirrored);
                result -= mirrored;
            } break;
        }

        texAccessor.set(i, channel, result * scale);
    }
}
*/

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main()
{
    SharedMemoryProxy smemAccessor;

    TextureProxy<CHANNELS> texAccessor;
    texAccessor.activeAxis = pc.flip ? uint16_t(1):uint16_t(0);
    texAccessor.load();

    // set us up
    BoxBlur<decltype(texAccessor),decltype(smemAccessor),WORKGROUP_SIZE> blur;
    blur.borderColor = float32_t3(0,1,0);
    blur.radius = pc.radius;
    blur.wrapMode = pc.edgeWrapMode;
    for (uint16_t ch=0; ch<CHANNELS; ch++)
    for (uint16_t pass=0; pass<PASSES; pass++)
    {
        // its the `smemAccessor` that gets aliased and reused so we need to barrier on its memory
        if (ch!=0 && pass!=0)
            smemAccessor.workgroupExecutionAndMemoryBarrier();
        blur(texAccessor,smemAccessor,ch);
    }

    texAccessor.store();
}
