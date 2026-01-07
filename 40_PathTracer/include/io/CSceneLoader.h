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

		// When outputFilePath isn't set in Film Element in Mitsuba, use this to find the extension string.
		static inline std::string_view fileExtensionFromFormat(ext::MitsubaLoader::CElementFilm::FileFormat format)
		{
			using FileFormat = ext::MitsubaLoader::CElementFilm::FileFormat;
			switch (format)
			{
				case FileFormat::PNG:
					return ".png";
				case FileFormat::OPENEXR:
					return ".exr";
				case FileFormat::JPEG:
					return ".jpg";
				default:
					break;
			}
			return "";
		}

		struct SLoadResult
		{
			struct SSensor
			{
				using type_e = ext::MitsubaLoader::CElementSensor::Type;

				inline SSensor() = default;
				inline SSensor(const SSensor&) = default;
				inline SSensor(SSensor&&) = default;
				inline SSensor& operator=(const SSensor&) = default;
				inline SSensor& operator=(SSensor&&) = default;

				inline operator bool() const
				{
					return bool(constants) && mutableDefaults.valid(constants) && bool(dynamicDefaults);
				}

				struct SConstants
				{
					constexpr static inline uint32_t MaxWidth = 0x1u<<(sizeof(uint16_t)*8-2);
					constexpr static inline uint32_t MaxHeight = MaxWidth;
					constexpr static inline uint32_t MaxCascadeCount = 15;

					inline operator bool() const
					{
						if (width <= 0 || width >= MaxWidth)
							return false;
						if (height <= 0 || height >= MaxHeight)
							return false;
						if (type != type_e::INVALID)
							return false;
						if (cascadeCount <= 0 || cascadeCount >= MaxCascadeCount)
							return false;
						return true;
					}

					// where the FFT bloom kernel is
					system::path bloomFilePath = {};
					//
					uint32_t width = 0u;
					uint32_t height = 0u;
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

					inline uint8_t getClipPlaneCount() const
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

					inline bool valid(const SConstants& cst) const
					{
						// TODO more checks
						return getClipPlaneCount()<MaxClipPlanes;
					}

					// inverse of view matrix, can include SCALE !
					hlsl::float32_t3x4 absoluteTransform;
					// if non-invertible, then spherical
					struct Raygen
					{
						// TODO: thin lens and telecentric support
						hlsl::float32_t4x4 linearProj = {};
					} raygen;
					//
					std::array<hlsl::float32_t4,MaxClipPlanes> clipPlanes = {};
					// denoiser and bloom require rendering with a "skirt" this controls the skirt size
					int32_t cropWidth = 0u;
					int32_t cropHeight = 0u;
					int32_t cropOffsetX = 0u;
					int32_t cropOffsetY = 0u;
					//
					float nearClip;
					float farClip;
					//
					float cascadeLuminanceBase = core::nan<float>();
					float cascadeLuminanceStart = core::nan<float>();
				} mutableDefaults = {};
				// these can change without having to reset accumulations, etc.
				struct SDynamic
				{
					// For a legacy `smgr->addCameraSceneNodeModifiedMaya(nullptr, -1.0f * mainSensorData.rotateSpeed, 50.0f, mainSensorData.moveSpeed, -1, 2.0f, defaultZoomSpeedMultiplier, false, true)`
					constexpr static inline float DefaultRotateSpeed = 300.0f;
					constexpr static inline float DefaultZoomSpeed = 1.0f;
					constexpr static inline float DefaultMoveSpeed = 100.0f;
					constexpr static inline float DefaultSceneSize = 50.0f; // reference for default zoom and move speed;
					// no constexpr std::pow
					//constexpr static inline float DefaultZoomSpeedMultiplier = std::pow(DefaultSceneSize,DefaultZoomSpeed/DefaultSceneSize);

					struct SPostProcess
					{
						float bloomScale = 0.0f;
						float bloomIntensity = 0.0f;
						std::string tonemapperArgs = "";
					};
					
					//
					inline operator bool() const
					{
						// TODO more checks
						return !hlsl::isnan(moveSpeed);
					}

					// members
					system::path outputFilePath = {};
					SPostProcess postProc = {};
					// even though spherical can't rotate, the preview camera can
					hlsl::float32_t3 up = {};
					float rotateSpeed = core::nan<float>();
					union
					{
						/*
		float linearStepZoomSpeed = sensorData.stepZoomSpeed;
		if(core::isnan<float>(sensorData.stepZoomSpeed))
		{
			linearStepZoomSpeed = sceneDiagonal * (DefaultZoomSpeed / DefaultSceneDiagonal);
		}

		// Set Zoom Multiplier
		{
			float logarithmicZoomSpeed = std::pow(sceneDiagonal, linearStepZoomSpeed / sceneDiagonal);
			sensorData.stepZoomSpeed =  logarithmicZoomSpeed;
			sensorData.getInteractiveCameraAnimator()->setStepZoomMultiplier(logarithmicZoomSpeed);
						*/
						struct SZoomable // spherical can't zoom
						{
							float speed = core::nan<float>();
						} zoomable = {};
					};
					//
					float moveSpeed = core::nan<float>();
					//
					uint32_t samplesNeeded = 0u;
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

#ifndef _NBL_THIS_EXAMPLE_C_SCENE_LOADER_CPP_
extern template struct nbl::system::impl::to_string_helper<nbl::this_example::CSceneLoader::SLoadResult::SSensor>;
#endif

#endif
