// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#ifndef _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_
#define _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_

#if 1
#include "nabla.h"
#else
#include "nbl/video/utilities/IUtilities.h"
#include "nbl/video/asset_traits.h"
#endif
#include "blake3.h"

namespace nbl::core
{
struct blake3_hash_t
{
	inline bool operator==(const blake3_hash_t&) const = default;

	uint8_t data[BLAKE3_OUT_LEN];
};

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

namespace std
{
template<>
struct hash<nbl::core::blake3_hash_t>
{
	inline size_t operator()(const nbl::core::blake3_hash_t& blake3) const
	{
		auto* as_p_uint64_t = reinterpret_cast<const size_t*>(blake3.data);
		size_t retval = as_p_uint64_t[0];
		for (auto i=1; i<BLAKE3_OUT_LEN; i++)
			retval ^= as_p_uint64_t[i] + 0x9e3779b97f4a7c15ull + (retval << 6) + (retval >> 2);
		return retval;
	}
};
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
		// meta tuple
		using supported_asset_types = std::tuple<
			asset::ICPUShader/*,
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout*/
		>;

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

		// This is the object we're trying to de-duplicate and hash-cons
		template<asset::Asset AssetType>
		struct asset_t
		{
			inline bool operator==(const asset_t<AssetType>& rhs) const = default;

			inline size_t hash() const
			{
				return ptrdiff_t(asset)^ptrdiff_t(uniqueCopyForUser);
			}

			inline bool valid() const
			{
				return asset;
			}

			const typename asset_traits<AssetType>::asset_t* asset = {};
			// Normally all references to the same IAsset* would spawn the same IBackendObject*, set this to non-null to override that behavior
			const asset::IAsset* uniqueCopyForUser = nullptr; // NOTE: this may have to move a bit farther down
		};
		// when getting dependents, these will be produced and patched appropriately
		template<asset::Asset AssetType>
		struct patch_t;
		template<>
		struct patch_t<asset::ICPUShader>
		{
			using this_t = patch_t<asset::ICPUShader>;

			inline patch_t() = default;
			inline patch_t(const asset::ICPUShader* shader) : stage(shader->getStage()) {}

			inline bool valid() const {return stage!=IGPUShader::ESS_UNKNOWN && stage!=IGPUShader::ESS_ALL_GRAPHICS;}

			inline this_t combine(const this_t& other) const
			{
				if (stage!=other.stage)
				{
					if (stage==IGPUShader::ESS_UNKNOWN)
						return other; // return the other whether valid or not
					else if (other.stage!=IGPUShader::ESS_UNKNOWN)
						return {}; // invalid
					// other is UNKNOWN so fallthrough and return us whether valid or not
				}
				return *this;
			}

			IGPUShader::E_SHADER_STAGE stage = IGPUShader::ESS_UNKNOWN;
		};
		template<>
		struct patch_t<asset::ICPUDescriptorSetLayout>
		{
			using this_t = patch_t<asset::ICPUDescriptorSetLayout>;

			inline patch_t() = default;
			inline patch_t(const asset::ICPUDescriptorSetLayout* layout) {}

			inline bool valid() const {return true;}

			inline this_t combine(const this_t& other) const
			{
				return *this;
			}
		};
		template<>
		struct patch_t<asset::ICPUPipelineLayout>
		{
			using this_t = patch_t<asset::ICPUPipelineLayout>;

			inline patch_t() = default;
			inline patch_t(const asset::ICPUPipelineLayout* pplnLayout)
			{
				const auto pc = pplnLayout->getPushConstantRanges();
				for (auto it=pc.begin(); it!=pc.end(); it++)
				for (auto byte=it->offset; byte<it->offset+it->size; byte++)
					pushConstantBytes[byte] = it->stageFlags;
			}

			inline bool valid() const {return true;}

			inline this_t combine(const this_t& other) const
			{
				this_t retval = *this;
				for (auto byte=0; byte!=pushConstantBytes.size(); byte++)
					retval.pushConstantBytes[byte] |= other.pushConstantBytes[byte];
				return retval;
			}

			std::array<core::bitflag<IGPUShader::E_SHADER_STAGE>,asset::CSPIRVIntrospector::MaxPushConstantsSize> pushConstantBytes = {IGPUShader::ESS_UNKNOWN};
		};
        //
        class CCache final : core::Uncopyable
        {
			public:
				inline CCache() = default;
				inline CCache(CCache&&) = default;
				inline ~CCache() = default;

				inline CCache& operator=(CCache&&) = default;

				// Typed Input (for a particular AssetType)
				template<asset::Asset AssetType>
				struct key_t
				{
					// constructor for when you want to override the creation parameters for the GPU Object while keeping the asset const
					inline key_t(const asset_t<AssetType>& _asset, const patch_t<AssetType>& _patch) : asset(_asset), patch(_patch) {}
					// the default constructor that deducts the patch params from a const asset which we don't want to change (so far)
					inline key_t(const asset_t<AssetType>& _asset) : asset(_asset), patch(asset.asset) {}

					inline bool operator==(const key_t<AssetType>&) const = default;

					inline bool valid() const
					{
						return asset.valid() && patch.valid();
					}

					asset_t<AssetType> asset;
					patch_t<AssetType> patch;
				};

#if 0
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
#endif

                template<asset::Asset AssetType>
                inline auto find(const key_t<AssetType>& key) const
                {
					core::blake3_hash_t hash;
					{
						blake3_hasher hasher;
						blake3_hasher_init(&hasher);
						blake3_hasher_update(hasher,&key.asset.uniqueCopyForUser,sizeof(key.asset.uniqueCopyForUser));
						fill_hash(&hasher,key.asset.asset);
						blake3_hasher_finalize(&hasher,hash.data,sizeof(hash));
					}
                    return find<AssetType>(hash);
                }
				// fastest lookup
                template<asset::Asset AssetType>
                inline typename asset_traits<AssetType>::video_t find(const core::blake3_hash_t& hash) const
                {
                    return std::get<cache_t<AssetType>>(m_caches).find(hash)->second.get();
                }

            private:
				//
				template<asset::Asset AssetType>
				struct cached
				{
					private:
						using video_t = typename asset_traits<AssetType>::video_t;
						constexpr static inline bool RefCtd = core::ReferenceCounted<video_t>;

					public:
						inline cached() = default;
						inline cached(const cached<AssetType>& other) : cached() {operator=(other);}
						inline cached(cached<AssetType>&&) = default;

						// special wrapping to make smart_refctd_ptr copyable
						inline cached<AssetType>& operator=(const cached<AssetType>& rhs)
						{
							if constexpr (RefCtd)
								value = core::smart_refctd_ptr<video_t>(rhs.value.get());
							else
								value = rhs;
							return *this;
						}
						inline cached<AssetType>& operator=(cached<AssetType>&&) = default;

						inline auto get() const
						{
							if constexpr (RefCtd)
								return value.get();
							else
								return value;
						}

						using type = std::conditional_t<RefCtd,core::smart_refctd_ptr<video_t>,video_t>;
						type value = {};
				};
				// The blake3 hash is quite fat (256bit), so we don't actually store a full asset ref for comparison.
				// Assuming a uniform distribution of keys and perfect hashing, we'd expect a collision on average every 2^256 asset loads.
				// Or if you actually calculate the P(X>1) for any reasonable number of asset loads (k trials), the Poisson CDF will be pratically 0.
				template<asset::Asset AssetType>
				using cache_t = std::unordered_map<core::blake3_hash_t,cached<AssetType>>;
				//
				using caches_t = core::tuple_transform_t<cache_t,supported_asset_types>;

				template<asset::Asset AssetType>
				static void fill_hash(blake3_hasher* hasher, const AssetType* asset, const patch_t<AssetType>& patch);

				//
 //               caches_t m_caches;
				cache_t<asset::ICPUShader> m_test;
        };

		// A meta class to encompass all the Assets you might want to convert at once
        struct SInputs
        {
			// Typed Range of Inputs of the same type
            template<asset::Asset AssetType>
            using span_t = std::span<const CCache::key_t<AssetType>>;

			// can be `nullptr` and even equal to `this`
			CAssetConverter* readCache = nullptr;

			// A type-sorted non-polymorphic list of "root assets"
			core::tuple_transform_t<span_t,supported_asset_types> assets = {};
        };
        struct SResults
        {
			public:
				inline SResults(SResults&&) = default;
				inline SResults(const SResults&) = delete;
				inline ~SResults() = default;
				inline SResults& operator=(const SResults&) = delete;
				inline SResults& operator=(SResults&&) = default;

				//
				inline bool reserveSuccess() const {return m_success;}

				// What queues you'll need to run the submit
				inline core::bitflag<IQueue::FAMILY_FLAGS> getRequiredQueueFlags() const {return m_queueFlags;}

				// for every entry in the input array, we 
				template<asset::Asset AssetType>
				using result_t = asset_traits<AssetType>::video_t;

#if 0
				// we convert to dummy by default
				virtual inline void wipeAsset(IAsset* _asset)
				{
					_asset->convertToDummyObject();
				}
#endif

			private:
				friend class CAssetConverter;

				inline SResults() = default;

				//
//				core::smart_refctd_ptr<> ; 
				//
				core::bitflag<IQueue::FAMILY_FLAGS> m_queueFlags = IQueue::FAMILY_FLAGS::NONE;
				//
				bool m_success = true;
        };
#define NBL_API
		// First Pass: Explore the DAG of Assets and "gather" patch infos for creating/retrieving equivalent GPU Objects.
		NBL_API SResults reserve(const SInputs& inputs);

		//
		struct SConvertParams
		{
			// by default the compute queue will own everything after all transfer operations are complete
			virtual inline SIntendedSubmitInfo& getFinalOwnerSubmit(IDeviceMemoryBacked* imageOrBuffer)
			{
				return compute;
			}

			// submits the buffered up cals 
			NBL_API ISemaphore::future_t<bool> autoSubmit();

			SIntendedSubmitInfo transfer = {};
			SIntendedSubmitInfo compute = {};
			IUtilities* utilities = nullptr;
		};
		// Second Pass: Actually create the GPU Objects
		NBL_API bool convert(SResults& reservations, SConvertParams& params);
#undef NBL_API

    protected:
        inline CAssetConverter(const SCreationParams& params) : m_params(std::move(params)) {}
        virtual inline ~CAssetConverter() = default;

        SCreationParams m_params;
};

}
#endif