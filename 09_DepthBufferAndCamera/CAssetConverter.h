// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#ifndef _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_
#define _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_

#include "nbl/video/utilities/IUtilities.h"
#include "nbl/video/asset_traits.h"


namespace nbl::core
{
// I only thought if I could, not if I should
template<template<class> class X, typename Tuple>
struct tuple_transform
{
	private:
		template<typename... T>
		static std::tuple<X<T>...> _impl(const std::tuple<T...>&);

	public:
		using type = decltype(_impl(std::declval<Tuple>()));
};
template<template<class> class X, typename Tuple>
using tuple_transform_t = tuple_transform<X,Tuple>::type;

//
template <class _Callable, class _Tuple>
constexpr void visit(_Callable&& _Obj, _Tuple& _Tpl) noexcept
{
	std::apply([&_Obj](auto& ...x){(..., _Obj(x));},_Tpl);
}
}

namespace nbl::video
{
/*
* This whole class assumes all assets you are converting will be used in read-only mode by the Device.
* It's a valid assumption for everything from pipelines to shaders, but not for descriptors (with exception of samplers) and sets.
* 
* Only Descriptors (by proxy their backing objects) and their Sets can have their contents changed after creation.
* 
* What this converter does is it computes hashes and compares equality based on the contents of an IAsset, not the pointer!
* With some sane limits, its not going to compare the contents of an ICPUImage or ICPUBuffer.
* 
* Therefore if you don't want some resource to be deduplicated you need to "explicitly" add it to `SInputs:assets` span.
* And if you want multiple references to the same `IAsset*` to convert to different Device Objects, you need to manually `clone()` them.
*/
class CAssetConverter : public core::IReferenceCounted
{
    public:
        struct SCreationParams
        {
            inline bool valid() const
            {
                if (!device)
                    return false;

                if (pipelineCache && pipelineCache->getOriginDevice()!=device) 
                    return false;

                return true;
            }

            // required not null
            ILogicalDevice* device = nullptr;
            // optional
			core::smart_refctd_ptr<const asset::ISPIRVOptimizer> optimizer = {};
			core::smart_refctd_ptr<asset::IShaderCompiler::CCache> compilerCache = {};
			core::smart_refctd_ptr<IGPUPipelineCache> pipelineCache = {};
        };
        static inline core::smart_refctd_ptr<CAssetConverter> create(SCreationParams&& params)
        {
            if (!params.valid())
                return nullptr;
            return core::smart_refctd_ptr<CAssetConverter>(new CAssetConverter(std::move(params)),core::dont_grab);
        }

        // meta tuple
        using supported_asset_types = std::tuple<
            asset::ICPUShader/*,
            asset::ICPUDescriptorSetLayout,
            asset::ICPUPipelineLayout*/
        >;
        //
        class CCache final
        {
			private:
                template<asset::Asset AssetType>
                struct search_t
                {
					const typename asset_traits<AssetType>::asset_t* asset = nullptr;
					std::optional<typename asset_traits<AssetType>::patch_t> patch = {};
                };
				template<asset::Asset AssetType>
				struct key_t
				{
					asset_traits<AssetType>::content_t content = {};
					asset_traits<AssetType>::patch_t patch = {};
				};

			public:
				struct Hash
				{
					template<asset::Asset AssetType>
					inline size_t operator()(const key_t<AssetType>& key) const
					{
						return 0x45; // TODO
					}
					template<asset::Asset AssetType>
					inline size_t operator()(const search_t<AssetType>& key) const
					{
						return 0x45; // TODO
					}
				};
				struct Equal
				{
					template<asset::Asset AssetType>
					inline bool operator()(const key_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return true; // TODO
					}
					template<asset::Asset AssetType>
					inline bool operator()(const key_t<AssetType>& lhs, const search_t<AssetType>& rhs) const
					{
						return true; // TODO
					}
				};
				template<asset::Asset AssetType>
				using cache_t = core::unordered_map<key_t<AssetType>,typename asset_traits<AssetType>::video_t,Hash,Equal>;

            public:
				inline void merge(const CCache& other)
				{
#if 0
					std::apply([&](auto&... caches)->void{
						(..., caches.merge(std::get<decltype(caches)>(other.m_caches)));
					},m_caches);
#endif
				}

				template<asset::Asset AssetType>
				inline auto end() const
				{
					return std::get<cache_t<AssetType>>(m_caches).end();
				}

                template<asset::Asset AssetType>
                inline auto find(const search_t<AssetType>& key, const size_t hashval) const
                {
                    return std::get<cache_t<AssetType>>(m_caches).find(key,hashval);
                }

            private:
				using caches_t = core::tuple_transform_t<cache_t,supported_asset_types>;

				//
                caches_t m_caches;
        };
		//a root asset we use for `SInput`
		template<asset::Asset AssetType>
		struct root_t
		{
			inline bool operator==(const root_t<AssetType>& rhs) const
			{
				return asset==rhs.asset && unique==rhs.unique;
			}

			const typename asset_traits<AssetType>::asset_t* asset = {};
			// whether to NOT compare equal with dedup hash later on
			bool unique = false;
		};
		// when getting dependents, these will be produced
		template<asset::Asset AssetType>
		struct input_t : root_t<AssetType>
		{
			std::optional<typename asset_traits<AssetType>::patch_t> patch = {};
		};
        struct SInput
        {
            template<asset::Asset AssetType>
            using span_t = std::span<const input_t<AssetType>>;
#if 0
			// we convert to dummy by default
			virtual inline void wipeAsset(IAsset* _asset)
			{
				_asset->convertToDummyObject();
			}
#endif
			// can be `nullptr` and even equal to `this`
			CAssetConverter* readCache = nullptr;

			core::tuple_transform_t<span_t,supported_asset_types> assets = {};
        };
        struct SResults
        {
				template<asset::Asset AssetType>
				using result_t = core::vector<typename asset_traits<AssetType>::video_t>;

			public:
				inline ~SResults() = default;

				// TODO
				inline bool reserveSuccess() const {return true;}

				//
				inline core::bitflag<IQueue::FAMILY_FLAGS> getRequiredQueueFlags() const {return m_queueFlags;}

				template<asset::Asset AssetType>
				inline std::span<const typename asset_traits<AssetType>::video_t> get() const
				{
					return std::get<result_t<AssetType>>(m_videoObjects)[0x45];
				}

			protected:
				friend class CAssetConverter;
				inline SResults() = default;

			private:
				template<asset::Asset AssetType>
				inline auto& get()
				{
					return std::get<result_t<AssetType>>(m_videoObjects);
				}

				// There's a 1:1 mapping between the `SInput::assets` and `videoObjects`
				// After returning from `reserve` and before entering `convert`
				core::tuple_transform_t<result_t,supported_asset_types> m_videoObjects = {};
				//
				core::bitflag<IQueue::FAMILY_FLAGS> m_queueFlags = IQueue::FAMILY_FLAGS::NONE;
        };
#define NBL_API
		NBL_API SResults reserve(const SInput& input);
		struct SConvertParams
		{
			// by default the compute queue will own everything
			virtual inline SIntendedSubmitInfo& getFinalOwnerSubmit(IDeviceMemoryBacked* imageOrBuffer)
			{
				return compute;
			}

			// submits the buffered up cals 
			virtual inline ISemaphore::future_t<bool> autoSubmit()
			{
			}

			SIntendedSubmitInfo transfer = {};
			SIntendedSubmitInfo compute = {};
			IUtilities* utilities = nullptr;
		};
		NBL_API bool convert(SResults& reservations, SConvertParams& params);
#undef NBL_API

    protected:
        inline CAssetConverter(const SCreationParams& params) : m_params(std::move(params)) {}
        virtual inline ~CAssetConverter() = default;

        SCreationParams m_params;
};

}
#endif