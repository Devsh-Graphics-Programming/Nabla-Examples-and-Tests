// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "nbl/builtin/hlsl/math/morton.hlsl"
#include <bitset>

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


// this time instead of defining our own `int main()` we derive from `nbl::system::IApplicationFramework` to play "nice" wil all platforms
class MortonTestApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = application_templates::MonoDeviceApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

		inline core::smart_refctd_ptr<video::IGPUShader> createShader(
			const char* includeMainName)
		{
			std::string prelude = "#include \"";
			auto CPUShader = core::make_smart_refctd_ptr<ICPUShader>((prelude + includeMainName + "\"\n").c_str(), IShader::E_SHADER_STAGE::ESS_COMPUTE, IShader::E_CONTENT_TYPE::ECT_HLSL, includeMainName);
			assert(CPUShader);
			return m_device->createShader(CPUShader.get());
		}
	public:
		MortonTestApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			createShader("app_resources/shader.hlsl");

			const auto masksArray = hlsl::morton::impl::decode_masks_array<uint32_t, 3>::Masks;
			for (auto i = 0u; i < 3; i++)
			{
				std::cout << std::bitset<32>(masksArray[i]) << std::endl;
			}

			return true;
		}

		// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
		void workLoopBody() override {}

		// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
		bool keepRunning() override {return false;}

	private:
		smart_refctd_ptr<nbl::video::CVulkanConnection> m_api;
};


NBL_MAIN_FUNC(MortonTestApp)