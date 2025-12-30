// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SCENE_LOADER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SCENE_LOADER_H_INCLUDED_


#include "nabla.h"
#include "nbl/builtin/hlsl/cpp_compat/promote.hlsl"

#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"


namespace nbl::this_example
{

class CSceneLoader : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		struct SCachedCreationParams
		{
			core::smart_refctd_ptr<asset::IAssetManager> assMan = nullptr;
			system::logger_opt_smart_ptr logger = nullptr;
		};
		struct SCreationParams : SCachedCreationParams
		{
			inline operator bool() const
			{
				if (!assMan)
					return false;
				return true;
			}
		};
		static core::smart_refctd_ptr<CSceneLoader> create(SCreationParams&& params);

		struct SLoadResult
		{
			struct SSensor
			{
				using type_e = ext::MitsubaLoader::CElementSensor::Type;

				struct SConstants
				{
					struct DenoiserArgs
					{
						// where the FFT bloom kernel is
						system::path bloomFilePath = {};
						float bloomScale = 0.0f;
						float bloomIntensity = 0.0f;
						std::string tonemapperArgs = "";
					};

					constexpr static inline uint32_t MaxWidth = 0x1u<<(sizeof(uint16_t)*8-2);
					constexpr static inline uint32_t MaxHeight = MaxWidth;
					constexpr static inline uint32_t MaxCascadeCount = 15;

					system::path outputFilePath = {};
					DenoiserArgs denoiserInfo = {};
					//
					uint32_t width = 0u;
					uint32_t height = 0u;
					// do we need to keep the crops?
					int32_t cropWidth = 0u;
					int32_t cropHeight = 0u;
					// could the offsets be dynamic ?
					int32_t cropOffsetX = 0u;
					int32_t cropOffsetY = 0u;
					//
					type_e type = type_e::INVALID;
					//
					uint8_t rightHandedCamera : 1 = true;
					uint8_t cascadeCount : 4 = 1;
				} constants = {};
				// these could theoretically change without recreating session resources
				struct SMutable
				{
					constexpr static inline uint8_t MaxClipPlanes = 6;

					inline uint8_t getClipPlaneCount()
					{
						using namespace nbl::hlsl;
						for (uint8_t i=0; i<MaxClipPlanes; i++)
						{
							const auto lhs = promote<float32_t3>(0.f);
							const auto& rhs = clipPlanes[i].xyz;
							if (any(glsl::notEqual<float32_t3>(lhs,rhs)))
								continue;
							return i;
						}
						return MaxClipPlanes;
					}

					//
					std::array<hlsl::float32_t4,MaxClipPlanes> clipPlanes = {};
					float cascadeLuminanceBase = core::nan<float>();
					float cascadeLuminanceStart = core::nan<float>();
				} mutableDefaults = {};
				// these can change without having to reset accumulations, etc.
				struct SDynamic
				{
					constexpr static inline float DefaultRotateSpeed = 300.0f;
					constexpr static inline float DefaultZoomSpeed = 1.0f;
					constexpr static inline float DefaultMoveSpeed = 100.0f;
					constexpr static inline float DefaultSceneDiagonal = 50.0f; // reference for default zoom and move speed;

					uint32_t samplesNeeded = 0u;
					float moveSpeed = core::nan<float>();
					float stepZoomSpeed = core::nan<float>();
					float rotateSpeed = core::nan<float>();
					float kappa = 0.f;
					float Emin = 0.05f;
				} dynamicDefaults = {};

			};

			inline operator bool() const
			{
				if (!scene || !sensors.empty())
					return false;
				return true;
			}

			//
			core::smart_refctd_ptr<const asset::ICPUScene> scene = {};
			//
			core::vector<SSensor> sensors;
			// TODO: for Material Compiler
			//std::future<bool> compileShadersFuture = {};
		};
		struct SLoadParams
		{
			system::path relPath = "";
			system::path workingDirectory = "";
		};
		SLoadResult load(SLoadParams&& _params);

    protected:
		struct SConstructorParams : SCachedCreationParams
		{
		};
		inline CSceneLoader(SConstructorParams&& _params) : m_params(std::move(_params)) {}
		virtual inline ~CSceneLoader() {}

		SConstructorParams m_params;
};

}
#endif
