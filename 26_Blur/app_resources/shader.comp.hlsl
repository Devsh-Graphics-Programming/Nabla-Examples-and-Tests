#include "nbl/builtin/hlsl/prefix_sum_blur/blur.hlsl"
#include "nbl/builtin/hlsl/prefix_sum_blur/box_sampler.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/colorspace/OETF.hlsl"
#include "common.hlsl"

using namespace nbl::hlsl;

uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(WORKGROUP_SIZE, 1, 1); }

[[vk::binding(0)]]
Texture2D<float32_t4> input;
[[vk::binding(1)]]
RWTexture2D<float32_t4> output;

[[vk::push_constant]] PushConstants pc;

template<uint16_t Chnls>
struct TextureProxy
{
    NBL_CONSTEXPR uint16_t Channels = Chnls;
    using texel_t = vector<float32_t, Channels>;

    // divisions by PoT constant will optimize out nicely
    template<typename T>
    T get(const uint16_t channel, const uint16_t uv)
    {
        return spill[uv / WORKGROUP_SIZE][channel];
    }

    template<typename T>
	void set(const uint16_t channel, const uint16_t uv, T value)
	{
        spill[uv / WORKGROUP_SIZE][channel] = value;
	}

    void load()
    {
        const uint16_t end = linearSize();
        uint16_t ix = workgroup::SubgroupContiguousIndex();
        // because workgroups do scans cooperatively all spill values need sane defaults
        for (uint16_t i=0; i < SpillSize; ix += WORKGROUP_SIZE)
            spill[i++] = ix < end ? (texel_t)input[position(ix)] : promote<texel_t>(0.f);
    }

    void store()
    {
        const uint16_t end = linearSize();
        uint16_t i = 0;
        // making sure that we don't store out of range
        for (uint16_t ix = workgroup::SubgroupContiguousIndex(); ix < end; ix += WORKGROUP_SIZE)
        {
            float32_t4 tmp = float32_t4(0, 0, 0, 1);
            for (uint16_t ch=0; ch < Channels; ch++)
                tmp[ch] = spill[i][ch];
            i++;
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
        pos[activeAxis] = ix;
        pos[activeAxis ^ 0x1] = _static_cast<uint16_t>(glsl::gl_WorkGroupID().x);
        return pos;
    }

    // whether we pas along X or Y
    uint16_t activeAxis;
    NBL_CONSTEXPR uint16_t SpillSize = (MAX_SCANLINE_SIZE - 1) / WORKGROUP_SIZE + 1;
    texel_t spill[SpillSize];
};

static const uint16_t MAX_SCAN_SCRATCH_SIZE = workgroup::scratch_size_arithmetic<WORKGROUP_SIZE, MAX_SUBGROUP_SIZE>::value + 2;

// we always use `uint32_t`
groupshared uint32_t smem[MAX_SCANLINE_SIZE];
groupshared uint32_t prefix_smem[MAX_SCAN_SCRATCH_SIZE];

struct SharedMemoryProxy
{
    NBL_CONSTEXPR uint16_t Size = MAX_SCANLINE_SIZE;

    template<typename T, typename I = uint16_t>
	enable_if_t<sizeof(T) == sizeof(uint32_t), T> get(const I idx)
	{
		return bit_cast<T>(smem[idx]);
	}

	template<typename T, typename I = uint16_t>
	enable_if_t<sizeof(T) == sizeof(uint32_t), void> set(const I idx, T value)
	{
        smem[idx] = bit_cast<uint32_t>(value);
	}

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }
};

struct ScanSharedMemoryProxy
{
    NBL_CONSTEXPR uint16_t Size = MAX_SCAN_SCRATCH_SIZE;

    template<typename T, typename I = uint16_t>
	enable_if_t<sizeof(T) == sizeof(uint32_t), void> get(const uint16_t idx, NBL_REF_ARG(T) val)
	{
		val = bit_cast<T>(prefix_smem[idx]);
	}

	template<typename T, typename I = uint16_t>
	enable_if_t<sizeof(T) == sizeof(uint32_t), void> set(const I idx, T value)
	{
        prefix_smem[idx] = bit_cast<uint32_t>(value);
	}

    void workgroupExecutionAndMemoryBarrier()
    {
        glsl::barrier();
    }
};

[numthreads(WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main()
{
    ScanSharedMemoryProxy scanSmemAccessor;

    TextureProxy<CHANNELS> texAccessor;
    texAccessor.activeAxis = (uint16_t)pc.activeAxis;
    texAccessor.load();
    
    prefix_sum_blur::BoxSampler<SharedMemoryProxy, float32_t> boxSampler;
    boxSampler.wrapMode = uint16_t(pc.edgeWrapMode);
    boxSampler.linearSize = texAccessor.linearSize();

    prefix_sum_blur::Blur1D<decltype(texAccessor), decltype(scanSmemAccessor), decltype(boxSampler), WORKGROUP_SIZE, jit::device_capabilities> blur;
    blur.radius = pc.radius;
    blur.borderColor = float32_t4(0, 1, 0, 1);

    for (uint16_t ch=0; ch < CHANNELS; ch++)
    for (uint16_t pass=0; pass < PASSES; pass++)
    {
        // its the `SharedMemoryProxy` that gets aliased and reused so we need to barrier on its memory
        if (ch != 0 && pass != 0)
            boxSampler.prefixSumAccessor.workgroupExecutionAndMemoryBarrier();
        blur(texAccessor, scanSmemAccessor, boxSampler, ch);
    }

    texAccessor.store();
}
