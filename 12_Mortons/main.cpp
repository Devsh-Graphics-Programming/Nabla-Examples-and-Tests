// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "app_resources/common.hlsl"
#include <bitset>

// Right now the test only checks that HLSL compiles the file
constexpr bool TestHLSL = true;

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
			{
				using namespace nbl::hlsl;

				auto bar = morton::code<false, 21, 3, emulated_uint64_t>::create(hlsl::vector<uint32_t, 3>(893728, 7843, 98032));
				auto foo = _static_cast<hlsl::vector<uint32_t, 3>>(bar);
				std::cout << foo[0] << " " << foo[1] << " " << foo[2] << " " << std::endl;
				
				//auto bar = morton::code<false, 21, 3, emulated_uint64_t>::create(hlsl::vector<uint32_t, 3>(893728, 7843, 98032));
				//std::cout << "High Encoded: " << std::bitset<32>(bar.value.data.x) << std::endl;
				//std::cout << "Low Encoded: " << std::bitset<32>(bar.value.data.y) << std::endl;
			}
			/*

			// ----------------------------------------------- CPP TESTS ----------------------------------------------------------------------
			
			// Coordinate extraction and whole vector decode tests
			{
				morton_t morton(vector_t(-1011, 765, 248));
				unsigned_morton_t unsignedMorton(unsigned_vector_t(154, 789, 1011));

				assert(morton.getCoordinate(0) == -1011 && morton.getCoordinate(1) == 765 && morton.getCoordinate(2) == 248);
				assert(unsignedMorton.getCoordinate(0) == 154u && unsignedMorton.getCoordinate(1) == 789u && unsignedMorton.getCoordinate(2) == 1011u);

				assert(static_cast<vector_t>(morton) == vector_t(-1011, 765, 248) && static_cast<unsigned_vector_t>(unsignedMorton) == unsigned_vector_t(154, 789, 1011));
			}

			// ***********************************************************************************************************************************
			// ************************************************* Arithmetic operator tests *******************************************************
			// ***********************************************************************************************************************************
			
			//  ----------------------------------------------------------------------------------------------------
			//  --------------------------------------- ADDITION ---------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// ---------------------------------------- Signed -----------------------------------------------------
			
			// No overflow
			assert(static_cast<vector_t>(morton_t(vector_t(-1011, 765, 248)) + morton_t(vector_t(1000, -985, 200))) == vector_t(-11, -220, 448));
			
			// Type 1 overflow: Addition of representable coordinates goes out of range
			assert(static_cast<vector_t>(morton_t(vector_t(-900, 70, 500)) + morton_t(vector_t(-578, -50, 20))) == vector_t(570, 20, -504));

			// Type 2 overflow: Addition of irrepresentable range gives correct result
			assert(static_cast<vector_t>(morton_t(vector_t(54, 900, -475)) + morton_t(vector_t(46, -1437, 699))) == vector_t(100, -537, 224));

			// ---------------------------------------- Unsigned -----------------------------------------------------

			// No overflow
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) + unsigned_morton_t(unsigned_vector_t(1563, 754, 220))) == unsigned_vector_t(1945, 1664, 763));

			// Type 1 overflow: Addition of representable coordinates goes out of range
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) + unsigned_morton_t(unsigned_vector_t(2000, 2000, 1000))) == unsigned_vector_t(334, 862, 519));

			// Type 2 overflow: Addition of irrepresentable range gives correct result
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) + unsigned_morton_t(unsigned_vector_t(-143, -345, -233))) == unsigned_vector_t(239, 565, 310));

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- SUBTRACTION -------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// ---------------------------------------- Signed -----------------------------------------------------

			// No overflow
			assert(static_cast<vector_t>(morton_t(vector_t(1000, 764, -365)) - morton_t(vector_t(834, -243, 100))) == vector_t(166, 1007, -465));

			// Type 1 overflow: Subtraction of representable coordinates goes out of range
			assert(static_cast<vector_t>(morton_t(vector_t(-900, 70, 500)) - morton_t(vector_t(578, -50, -20))) == vector_t(570, 120, -504));

			// Type 2 overflow: Subtraction of irrepresentable range gives correct result
			assert(static_cast<vector_t>(morton_t(vector_t(54, 900, -475)) - morton_t(vector_t(-46, 1437, -699))) == vector_t(100, -537, 224));

			// ---------------------------------------- Unsigned -----------------------------------------------------

			// No overflow
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) - unsigned_morton_t(unsigned_vector_t(322, 564, 299))) == unsigned_vector_t(60, 346, 244));

			// Type 1 overflow: Subtraction of representable coordinates goes out of range
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(382, 910, 543)) - unsigned_morton_t(unsigned_vector_t(2000, 2000, 1000))) == unsigned_vector_t(430, 958, 567));

			// Type 2 overflow: Subtraction of irrepresentable range gives correct result
			assert(static_cast<unsigned_vector_t>(unsigned_morton_t(unsigned_vector_t(54, 900, 475)) - unsigned_morton_t(unsigned_vector_t(-865, -100, -10))) == unsigned_vector_t(919, 1000, 485));


			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- UNARY NEGATION ----------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Only makes sense for signed
			assert(static_cast<vector_t>(- morton_t(vector_t(-1024, 543, -475))) == vector_t(-1024, -543, 475));

			// ***********************************************************************************************************************************
			// ************************************************* Comparison operator tests *******************************************************
			// ***********************************************************************************************************************************

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- OPERATOR< ---------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Signed
			
			// Same sign, negative
			assert(morton_t(vector_t(-954, -455, -333)) < morton_t(vector_t(-433, -455, -433)) == bool_vector_t(true, false, false));
			// Same sign, positive
			assert(morton_t(vector_t(954, 455, 333)) < morton_t(vector_t(433, 455, 433)) == bool_vector_t(false, false, true));
			// Differing signs
			assert(morton_t(vector_t(954, -32, 0)) < morton_t(vector_t(-44, 0, -1)) == bool_vector_t(false, true, false));

			// Unsigned
			assert(unsigned_morton_t(unsigned_vector_t(239, 435, 66)) < unsigned_morton_t(unsigned_vector_t(240, 435, 50)) == bool_vector_t(true, false, false));

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- OPERATOR<= --------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Signed

			// Same sign, negative
			assert(morton_t(vector_t(-954, -455, -333)) <= morton_t(vector_t(-433, -455, -433)) == bool_vector_t(true, true, false));
			// Same sign, positive
			assert(morton_t(vector_t(954, 455, 333)) <= morton_t(vector_t(433, 455, 433)) == bool_vector_t(false, true, true));
			// Differing signs
			assert(morton_t(vector_t(954, -32, 0)) <= morton_t(vector_t(-44, 0, -1)) == bool_vector_t(false, true, false));

			// Unsigned
			assert(unsigned_morton_t(unsigned_vector_t(239, 435, 66)) <= unsigned_morton_t(unsigned_vector_t(240, 435, 50)) == bool_vector_t(true, true, false));

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- OPERATOR> ---------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Signed

			// Same sign, negative
			assert(morton_t(vector_t(-954, -455, -333)) > morton_t(vector_t(-433, -455, -433)) == bool_vector_t(false, false, true));
			// Same sign, positive
			assert(morton_t(vector_t(954, 455, 333)) > morton_t(vector_t(433, 455, 433)) == bool_vector_t(true, false, false));
			// Differing signs
			assert(morton_t(vector_t(954, -32, 0)) > morton_t(vector_t(-44, 0, -1)) == bool_vector_t(true, false, true));

			// Unsigned
			assert(unsigned_morton_t(unsigned_vector_t(239, 435, 66)) > unsigned_morton_t(unsigned_vector_t(240, 435, 50)) == bool_vector_t(false, false, true));

			//  ----------------------------------------------------------------------------------------------------
			//  -------------------------------------- OPERATOR>= --------------------------------------------------
			//  ----------------------------------------------------------------------------------------------------

			// Signed

			// Same sign, negative
			assert(morton_t(vector_t(-954, -455, -333)) >= morton_t(vector_t(-433, -455, -433)) == bool_vector_t(false, true, true));
			// Same sign, positive
			assert(morton_t(vector_t(954, 455, 333)) >= morton_t(vector_t(433, 455, 433)) == bool_vector_t(true, true, false));
			// Differing signs
			assert(morton_t(vector_t(954, -32, 0)) >= morton_t(vector_t(-44, 0, -1)) == bool_vector_t(true, false, true));

			// Unsigned
			assert(unsigned_morton_t(unsigned_vector_t(239, 435, 66)) >= unsigned_morton_t(unsigned_vector_t(240, 435, 50)) == bool_vector_t(false, true, true));

			*/

			return true;
		}

		// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
		void workLoopBody() override {}

		// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
		bool keepRunning() override {return false;}

		// Cleanup
		bool onAppTerminated() override
		{
			return device_base_t::onAppTerminated();
		}

	private:
		smart_refctd_ptr<IGPUComputePipeline> m_pipeline;

		smart_refctd_ptr<nbl::video::IUtilities> m_utils;

		StreamingTransientDataBufferMT<>* m_downStreamingBuffer;
		smart_refctd_ptr<nbl::video::IGPUBuffer> m_deviceLocalBuffer;

		// These are Buffer Device Addresses
		uint64_t m_downStreamingBufferAddress;
		uint64_t m_deviceLocalBufferAddress;

		uint32_t m_alignment;

		smart_refctd_ptr<ISemaphore> m_timeline;
		uint64_t semaphorValue = 0;
};


NBL_MAIN_FUNC(MortonTestApp)