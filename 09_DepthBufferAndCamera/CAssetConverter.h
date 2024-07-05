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
#include "nbl/builtin/hlsl/cpp_compat.hlsl"


namespace nbl::video
{
/*
* This whole class assumes all assets you are converting will be used in read-only mode by the Device.
* It's a valid assumption for everything from pipelines to shaders, but not for descriptors (with exception of samplers) and their sets.
* 
* Only Descriptors (by proxy their backing objects) and their Sets can have their contents changed after creation.
* 
* What this converter does is it computes hashes and compares equality based on the contents of an IAsset, not the pointer!
* With some sane limits, its not going to compare the contents of an ICPUImage or ICPUBuffer.
* 
* Therefore if you don't want some resource to be deduplicated you need to "explicitly" let us know via `SInputs::getDependantUniqueCopyGroupID`.
*/
class CAssetConverter : public core::IReferenceCounted
{
	public:
		// meta tuple
		// TODO: how to make MSVC shut up about warning C4624 about deleted dtors? Use another container?
		using supported_asset_types = core::type_list<
			asset::ICPUShader/*,
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout*/,
			asset::ICPUBuffer/*,
			asset::ICPUBufferView*/
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

		// when getting dependents, the creation parameters of GPU objects will be produced and patched appropriately
		template<asset::Asset AssetType>
		struct patch_t;
		template<>
		struct patch_t<asset::ICPUShader>
		{
			using this_t = patch_t<asset::ICPUShader>;

			inline patch_t() = default;
			inline patch_t(const asset::ICPUShader* shader, const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits);

			inline bool valid() const {return nbl::hlsl::bitCount<uint32_t>(stage)!=1;}

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
			inline patch_t(const asset::ICPUDescriptorSetLayout* layout, const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits) {}

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
			patch_t(const asset::ICPUPipelineLayout* pplnLayout, const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits);

			inline bool valid() const {return !invalid;}

			inline this_t combine(const this_t& other) const
			{
				if (invalid || other.invalid)
					return {};
				this_t retval = *this;
				for (auto byte=0; byte!=pushConstantBytes.size(); byte++)
					retval.pushConstantBytes[byte] |= other.pushConstantBytes[byte];
				return retval;
			}

			std::array<core::bitflag<IGPUShader::E_SHADER_STAGE>,asset::CSPIRVIntrospector::MaxPushConstantsSize> pushConstantBytes = {IGPUShader::ESS_UNKNOWN};
			bool invalid = true;
		};
		template<>
		struct patch_t<asset::ICPUBuffer>
		{
			using this_t = patch_t<asset::ICPUBuffer>;
			using usage_flags_t = asset::IBuffer::E_USAGE_FLAGS;

			inline patch_t() = default;
			patch_t(const asset::ICPUBuffer* buffer, const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits);

			inline bool valid() const {return usage!=IGPUBuffer::E_USAGE_FLAGS::EUF_NONE;}

			inline this_t combine(const this_t& other)
			{
				usage |= other.usage;
				return *this;
			}

			core::bitflag<usage_flags_t> usage = usage_flags_t::EUF_NONE;
		};
		template<>
		struct patch_t<asset::ICPUSampler>
		{
			using this_t = patch_t<asset::ICPUSampler>;

			inline patch_t() = default;
			patch_t(const asset::ICPUSampler* sampler, const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits);

			inline bool valid() const {return anisotropyLevelLog2>5;}

			inline this_t combine(const this_t& other) const
			{
				// The only reason why someone would have a different level to creation parameters is
				// because the HW doesn't support that level and the level gets clamped. So must be same.
				if (anisotropyLevelLog2!=other.anisotropyLevelLog2)
					return {}; // invalid
				return *this;
			}

			uint8_t anisotropyLevelLog2 = 6;
		};
		// And these are the results of the conversion.
		template<asset::Asset AssetType>
		struct cached_t final
		{
			private:
				using this_t = cached_t<AssetType>;
				using video_t = typename asset_traits<AssetType>::video_t;
				constexpr static inline bool RefCtd = core::ReferenceCounted<video_t>;

			public:
				inline cached_t() = default;
				inline cached_t(const this_t& other) : cached_t() {operator=(other);}
				inline cached_t(this_t&&) = default;

				// special wrapping to make smart_refctd_ptr copyable
				inline this_t& operator=(const this_t& rhs)
				{
					if constexpr (RefCtd)
						value = core::smart_refctd_ptr<video_t>(rhs.value.get());
					else
						value = video_t(rhs.value);
					return *this;
				}
				inline this_t& operator=(this_t&&) = default;

				inline const auto& get() const
				{
					if constexpr (RefCtd)
						return value.get();
					else
						return value;
				}

				using type = std::conditional_t<RefCtd,core::smart_refctd_ptr<video_t>,video_t>;
				type value = {};
		};
#define NBL_API
		// Typed Cache (for a particular AssetType)
		template<asset::Asset AssetType>
        class CCache final : core::Uncopyable
        {
			public:
				inline CCache() = default;
				inline CCache(CCache&&) = default;
				inline ~CCache() = default;

				inline CCache& operator=(CCache&&) = default;

				// fastest lookup
				inline const auto& find(const core::blake3_hash_t& hash) const
				{
					return m_container.find(hash)->second.get();
				}

				struct lookup_t final
				{
					public:
						inline bool operator==(const lookup_t& rhs) const = default;

						inline core::blake3_hash_t hash() const
						{
							core::blake3_hash_t retval;
							{
								blake3_hasher hasher;
								blake3_hasher_init(&hasher);
								// We purposefully don't hash asset pointer, we hash the contents instead
								//core::blake3_hasher_update(hasher,asset);
								core::blake3_hasher_update(hasher,uniqueCopyGroupID);
								hash_impl(&hasher);
								blake3_hasher_finalize(&hasher,retval.data,sizeof(retval));
							}
							return retval;
						}

						inline bool valid() const
						{
							return asset && patch.valid();
						}

						const typename asset_traits<AssetType>::asset_t* asset = nullptr;
						// Normally all references to the same IAsset* would spawn the same IBackendObject*.
						// However, each unique integer value "spawns" a new copy, note that the group ID is the same size as a pointer, so you can e.g.
						// cast a pointer of the user (parent reference) to size_t and use that for a unique copy for the user.
						size_t uniqueCopyGroupID = 0;
						patch_t<AssetType> patch = {};

					private:
						NBL_API void hash_impl(blake3_hasher* hasher) const;
				};
				// slower but more useful version
                inline auto find(const lookup_t& key) const
                {
					if (key.valid())
						return m_container.end();

                    return find(key.hash());
                }

				inline void merge(const CCache<AssetType>& other)
				{
					m_container.insert(other.m_container.begin(),other.m_container.end());
				}

            private:
				// The blake3 hash is quite fat (256bit), so we don't actually store a full asset ref for comparison.
				// Assuming a uniform distribution of keys and perfect hashing, we'd expect a collision on average every 2^256 asset loads.
				// Or if you actually calculate the P(X>1) for any reasonable number of asset loads (k trials), the Poisson CDF will be pratically 0.
				std::unordered_map<core::blake3_hash_t,cached_t<AssetType>> m_container;
        };

		// A meta class to encompass all the Assets you might want to convert at once
        struct SInputs
        {
			struct instance_t
			{
				const asset::IAsset* asset = nullptr;
				size_t uniqueCopyGroupID = 0;
			};
			// You need to tell us if an asset needs multiple copies, separate for each user. The return value of this function dictates
			// what copy of the asset each user gets. Note that we also call it with `user=={nullptr,0}` for each entry in `SInputs::assets`.
			// NOTE: You might get extra copies within the same group ID due to inability to patch entries
			virtual inline size_t getDependantUniqueCopyGroupID(const instance_t& user, const asset::IAsset* dependant) const
			{
				return 0;
			}

			// Typed Range of Inputs of the same type
            template<asset::Asset AssetType>
            using asset_span_t = std::span<const typename asset_traits<AssetType>::asset_t* const>;
            template<asset::Asset AssetType>
            using patch_span_t = std::span<const patch_t<AssetType>>;

			// can be `nullptr` and even equal to `this`
			CAssetConverter* readCache = nullptr;

			// A type-sorted non-polymorphic list of "root assets"
			core::tuple_transform_t<asset_span_t,supported_asset_types> assets = {};
			// Optional: Whatever is not in `patches` will generate a default patch
			core::tuple_transform_t<patch_span_t,supported_asset_types> patches = {};
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

				// for every entry in the input array, we have this mapped 1:1
				template<asset::Asset AssetType>
				using vector_t = core::vector<cached_t<AssetType>>;

				// until `convert` is called, this will only contain valid entries for items already found in `SInput::readCache`
				template<asset::Asset AssetType>
				std::span<const cached_t<AssetType>> getGPUObjects() const {return std::get<vector_t<AssetType>>(m_gpuObjects);}

			private:
				friend class CAssetConverter;

				inline SResults() = default;

				//
				core::tuple_transform_t<vector_t,supported_asset_types> m_gpuObjects = {};
				//
//				m_reservations;
				//
				core::bitflag<IQueue::FAMILY_FLAGS> m_queueFlags = IQueue::FAMILY_FLAGS::NONE;
				//
				bool m_success = true;
        };
		// First Pass: Explore the DAG of Assets and "gather" patch infos for creating/retrieving equivalent GPU Objects.
		NBL_API SResults reserve(const SInputs& inputs);

		//
		struct SConvertParams
		{
			// By default the compute queue will own everything after all transfer operations are complete.
			virtual inline SIntendedSubmitInfo& getFinalOwnerSubmit(IDeviceMemoryBacked* imageOrBuffer, const core::blake3_hash_t& createdFrom)
			{
				return compute;
			}

			// submits the buffered up cals 
			NBL_API ISemaphore::future_t<bool> autoSubmit();

			// can be `nullptr`, but for most usages should be equal to `this`
			CAssetConverter* writeCache = nullptr;
			SIntendedSubmitInfo transfer = {};
			SIntendedSubmitInfo compute = {};
			IUtilities* utilities = nullptr;
			IGPUPipelineCache* pipelineCache = nullptr;
		};
		// Second Pass: Actually create the GPU Objects
		NBL_API bool convert(SResults& reservations, SConvertParams& params);
#undef NBL_API

		// Only const methods so others are not able to insert things made by e.g. different devices
		template<asset::Asset AssetType>
		inline const CCache<AssetType>& getCache() const
		{
			return std::get<CCache<AssetType>>(m_caches);
		}

		//
		inline void merge(const CAssetConverter* other)
		{
			std::apply([&](auto&... caches)->void{
				(..., caches.merge(std::get<std::remove_reference_t<decltype(caches)>>(other->m_caches)));
			},m_caches);
		}

    protected:
        inline CAssetConverter(const SCreationParams& params) : m_params(std::move(params)) {}
        virtual inline ~CAssetConverter() = default;

        SCreationParams m_params;
		core::tuple_transform_t<CCache,supported_asset_types> m_caches;
};

}

namespace std
{
template<>
struct hash<nbl::video::CAssetConverter::SInputs::instance_t>
{
	inline size_t operator()(const nbl::video::CAssetConverter::SInputs::instance_t& record) const noexcept
	{
		return ptrdiff_t(record.asset)^ptrdiff_t(record.uniqueCopyGroupID);
	}
};
}
#endif