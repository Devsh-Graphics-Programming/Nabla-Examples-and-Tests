#include <limits>

#include <nabla.h>

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::video;
using namespace nbl::application_templates;

static inline const nbl::video::ILogicalDevice* g_device = nullptr;

template<nbl::core::StringLiteral Expected, class Buffer>
constexpr bool buffer_equals(const Buffer& buf)
{
	constexpr size_t expected_size = sizeof(Expected.value) - 1;
	if (buf.size() != expected_size)
		return false;
	for (size_t i = 0; i < expected_size; ++i)
	{
		if (buf.data()[i] != Expected.value[i])
			return false;
	}
	return true;
}

template<nbl::core::StringLiteral Key, nbl::core::StringLiteral Expected, auto... Args>
bool check_key()
{
	constexpr auto keyBuf = nbl::this_example::builtin::build::get_spirv_key<Key>(Args...);
	constexpr bool matches_ct = buffer_equals<Expected>(keyBuf);
	static_assert(matches_ct);

	if constexpr (requires { nbl::core::detail::SpirvKeyBuilder<Key>::build_from_device(static_cast<const nbl::video::ILogicalDevice*>(nullptr), Args...); })
	{
		if (g_device)
			return buffer_equals<Expected>(nbl::this_example::builtin::build::get_spirv_key<Key>(g_device, Args...));
	}

	return buffer_equals<Expected>(nbl::this_example::builtin::build::get_spirv_key<Key>(Args...));
}

static constexpr struct
{
	bool b1 = true;
	bool b0 = false;
	uint16_t u16 = 65535u;
	uint32_t u32 = 123456789u;
	uint64_t u64 = 1234567890123ull;
	int16_t s16 = -32768;
	int32_t s32 = -123456789;
	int64_t s64 = -1234567890123ll;
} userI;

static constexpr struct
{
	uint32_t sel = 1u;
} userT;

static constexpr struct
{
	float min = std::numeric_limits<float>::min();
	float max = std::numeric_limits<float>::max();
	float neg = -1.0f;
	float exp = 1.25e-1f;
} userF;

static constexpr struct
{
	double min = std::numeric_limits<double>::min();
	double max = std::numeric_limits<double>::max();
	double neg = -1.0;
	double exp = 1.25e-1;
} userD;

static constexpr struct
{
	uint32_t md = 7u;
	bool en = true;
	float sc = 1.0f;
} userMix;

static constexpr struct
{
	uint32_t maxImageDimension2D = 16384u;
} limits;

static constexpr struct
{
	bool shaderCullDistance = true;
} features;

class SpirvKeysTestApp final : public MonoDeviceApplication
{
	using device_base_t = MonoDeviceApplication;

public:
	SpirvKeysTestApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		g_device = m_device.get();

		bool ok = true;
#define SPIRV_KEY_TEST(KEY, EXPECTED, ...) ok &= check_key<KEY, EXPECTED __VA_OPT__(,) __VA_ARGS__>();
		SPIRV_KEY_TEST("k_plain", "Debug/k_plain.spv")
		SPIRV_KEY_TEST("k_ints", "Debug/k_ints__i.b1_1.b0_0.u16_65535.u32_123456789.u64_1234567890123.s16_-32768.s32_-123456789.s64_-1234567890123.spv", userI)
		SPIRV_KEY_TEST("k_two", "Debug/k_two__t.sel_1.spv", userT)
		SPIRV_KEY_TEST("k_f", "Debug/k_f__f.min_1.17549435e-38.max_3.40282347e+38.neg_-1.00000000e+00.exp_1.25000000e-01.spv", userF)
		SPIRV_KEY_TEST("k_d", "Debug/k_d__d.min_2.2250738585072010e-308.max_1.7976931348623165e+308.neg_-1.0000000000000000e+00.exp_1.2500000000000000e-01.spv", userD)
		SPIRV_KEY_TEST("k_mix", "Debug/k_mix__m.md_7.en_1.sc_1.00000000e+00__limits.maxImageDimension2D_16384__features.shaderCullDistance_1.spv", userMix, limits, features)
#undef SPIRV_KEY_TEST

		if (!ok)
			return logFail("SpirvKeysTest failed");

		return true;
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(SpirvKeysTestApp)
