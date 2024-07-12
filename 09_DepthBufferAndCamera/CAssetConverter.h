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

#if 0
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
					std::apply([&](auto&... caches)->void{
						(..., caches.merge(std::get<decltype(caches)>(other.m_caches)));
					},m_caches);
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
#endif
		// This is the object we're trying to de-duplicate and hash-cons
		template<asset::Asset AssetType>
		struct key_t
		{
			inline bool operator==(const key_t<AssetType>& rhs) const
			{
				return asset==rhs.asset && unique==rhs.unique;
			}

			const typename asset_traits<AssetType>::asset_t* asset = {};
			// whether to NOT compare equal with dedup hash later on
			bool unique = false;
		};
		// when getting dependents, these will be produced
		template<asset::Asset AssetType>
		struct patch_t;
		template<>
		struct patch_t<asset::ICPUShader>
		{
			inline patch_t(const asset::ICPUShader* shader=nullptr) {}
		};
		template<>
		struct patch_t<asset::ICPUDescriptorSetLayout>
		{
			inline patch_t(const asset::ICPUDescriptorSetLayout* layout=nullptr) {}
		};
		template<>
		struct patch_t<asset::ICPUPipelineLayout>
		{
			inline patch_t(const asset::ICPUPipelineLayout* pplnLayout=nullptr)
			{
			}

			// TODO: unique form of push constant ranges
		};
		// meta tuple
		using supported_asset_types = std::tuple<
			asset::ICPUShader/*,
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout*/
		>;
        struct SInput
        {
			template<asset::Asset AssetType>
			struct input_t
			{
				inline input_t(const key_t<AssetType>& _key, const patch_t<AssetType>& _patch) : key(_key), patch(_patch) {}
				inline input_t(const key_t<AssetType>& _key) : key(_key), patch(key) {}

				key_t<AssetType> key;
				patch_t<AssetType> patch;
			};
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
			public:
				inline ~SResults() = default;

				//
				inline bool reserveSuccess() const {return m_success;}

				//
				inline core::bitflag<IQueue::FAMILY_FLAGS> getRequiredQueueFlags() const {return m_queueFlags;}


			protected:
				inline SResults() = default;


			private:
				friend class CAssetConverter;

				template<asset::Asset AssetType>
				struct result_t
				{
					inline const auto& get() const
					{
						if (canonical)
							return canonical->result;
						return result;
					}

					patch_t<AssetType> patch = {};
					const result_t<AssetType>* canonical = nullptr;
					asset_traits<AssetType>::video_t result = {};
				};
				//
				struct key_hash
				{
					template<asset::Asset AssetType>
					inline size_t operator()(const key_t<AssetType>& in) const
					{
						return std::hash<const void*>{}(in.asset)^(in.unique ? (~0x0ull):0x0ull);
					}
				};
				template<asset::Asset AssetType>
				using dag_cache_t = core::unordered_multimap<key_t<AssetType>,result_t<AssetType>,key_hash>;
				
				core::tuple_transform_t<dag_cache_t,supported_asset_types> m_typedDagNodes = {};
				//
				core::bitflag<IQueue::FAMILY_FLAGS> m_queueFlags = IQueue::FAMILY_FLAGS::NONE;
				//
				bool m_success = true;
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