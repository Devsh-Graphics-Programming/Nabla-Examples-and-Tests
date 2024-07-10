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
		using supported_asset_types = core::type_list<
			asset::ICPUShader,
			asset::ICPUBuffer,
			asset::ICPUSampler,
			// image,
//			asset::ICPUBufferView,
			// image view
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout
		>;

		struct SCreationParams
		{
			inline bool valid() const
			{
				if (!device)
					return false;

				return true;
			}

			// required not null
			ILogicalDevice* device = nullptr;
			// optional
			core::smart_refctd_ptr<const asset::ISPIRVOptimizer> optimizer = {};
		};
		static inline core::smart_refctd_ptr<CAssetConverter> create(SCreationParams&& params)
		{
			if (!params.valid())
				return nullptr;
			return core::smart_refctd_ptr<CAssetConverter>(new CAssetConverter(std::move(params)),core::dont_grab);
		}

		// When getting dependents, the creation parameters of GPU objects will be produced and patched appropriately.
		// `patch_t` uses CRPT to inherit from `patch_impl_t` to provide default `operator==` and `update_hash()` definition.
		// The default specialization kicks in for any `AssetType` that has nothing possible to patch (e.g. Descriptor Set Layout).
		template<asset::Asset AssetType>
		struct patch_impl_t
		{
#define PATCH_IMPL_BOILERPLATE(ASSET_TYPE,...) using this_t = patch_impl_t<ASSET_TYPE>; \
			public: \
				inline patch_impl_t() = default; \
				inline patch_impl_t(const this_t& other) = default; \
				inline patch_impl_t(this_t&& other) = default; \
				inline this_t& operator=(const this_t& other) = default; \
				inline this_t& operator=(this_t&& other) = default; \
				__VA_ARGS__ patch_impl_t(const ASSET_TYPE* asset, const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits)

				PATCH_IMPL_BOILERPLATE(AssetType,inline) {}

				// always valid
				inline bool valid() const {return true;}

			protected:
				// there's nothing to combine, so combining always produces the input
				inline this_t combine(const this_t& other) const
				{
					return *this;
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUShader>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUShader);

				inline bool valid() const {return nbl::hlsl::bitCount<uint32_t>(stage)!=1;}

				IGPUShader::E_SHADER_STAGE stage = IGPUShader::ESS_UNKNOWN;

			protected:
				inline this_t combine(const this_t& other) const
				{
					if (stage != other.stage)
					{
						if (stage == IGPUShader::ESS_UNKNOWN)
							return other; // return the other whether valid or not
						else if (other.stage != IGPUShader::ESS_UNKNOWN)
							return {}; // invalid
						// other is UNKNOWN so fallthrough and return us whether valid or not
					}
					return *this;
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUBuffer>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUBuffer);

				inline bool valid() const {return usage!=IGPUBuffer::E_USAGE_FLAGS::EUF_NONE;}

				using usage_flags_t = IGPUBuffer::E_USAGE_FLAGS;
				core::bitflag<usage_flags_t> usage = usage_flags_t::EUF_NONE;

			protected:
				inline this_t combine(const this_t& other) const
				{
					this_t retval = *this;
					retval.usage |= other.usage;
					return *this;
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUSampler>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUSampler);

				inline bool valid() const { return anisotropyLevelLog2>5; }

				uint8_t anisotropyLevelLog2 = 6;
				
			protected:
				inline this_t combine(const this_t& other) const
				{
					// The only reason why someone would have a different level to creation parameters is
					// because the HW doesn't support that level and the level gets clamped. So must be same.
					if (anisotropyLevelLog2 != other.anisotropyLevelLog2)
						return {}; // invalid
					return *this;
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUPipelineLayout>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUPipelineLayout);

				inline bool valid() const {return !invalid;}

				std::array<core::bitflag<IGPUShader::E_SHADER_STAGE>,asset::CSPIRVIntrospector::MaxPushConstantsSize> pushConstantBytes = {IGPUShader::ESS_UNKNOWN};
				bool invalid = true;
				
			protected:
				inline this_t combine(const this_t& other) const
				{
					if (invalid || other.invalid)
						return {};
					this_t retval = *this;
					for (auto byte=0; byte!=pushConstantBytes.size(); byte++)
						retval.pushConstantBytes[byte] |= other.pushConstantBytes[byte];
					return retval;
				}
		};
#undef PATCH_IMPL_BOILERPLATE
		// The default specialization provides simple equality operations and hash operations, this will work as long as your patch_impl_t doesn't:
		// - use a container like `core::vector<T>`, etc.
		// - use pointers to other objects or arrays whose contents must be analyzed
		template<asset::Asset AssetType>
		struct patch_t : patch_impl_t<AssetType>
		{
			using this_t = patch_t<AssetType>;
			using base_t = patch_impl_t<AssetType>;

			// forwarding
			using base_t::base_t;
			inline patch_t(const this_t& other) : base_t(other) {}
			inline patch_t(this_t&& other) : base_t(std::move(other)) {}
			inline patch_t(base_t&& other) : base_t(std::move(other)) {}

			inline this_t& operator=(const this_t& other) = default;
			inline this_t& operator=(this_t&& other) = default;

			// "unhide" this method
			inline this_t combine(const this_t& other) const
			{
				return base_t::combine(other);
			}

			// actual new methods
			inline bool operator==(const patch_t<AssetType>& other) const
			{
				return memcmp(this,&other,sizeof(base_t))==0;
			}

			inline void hasher_update(::blake3_hasher& hasher) const
			{
				core::blake3_hasher_update(hasher,*this);
			}
		};
#define NBL_API
		// A class to accelerate our hash computations
		class CHashCache final : core::IReferenceCounted
		{
			public:
				//
				template<asset::Asset AssetType>
				struct lookup_t
				{
					inline bool valid() const
					{
						return asset && patch && patch->valid();
					}

					const AssetType* asset = nullptr;
					// Normally all references to the same IAsset* would spawn the same IBackendObject*.
					// However, each unique integer value "spawns" a new copy, note that the group ID is the same size as a pointer, so you can e.g.
					// cast a pointer of the user (parent reference) to size_t and use that for a unique copy for the user.
					size_t uniqueCopyGroupID = 0;
					const patch_t<AssetType>* patch = {};
					// how deep from `asset` do we start trusting the cache to contain correct non stale hashes
					uint32_t hashTrustLevel = 0;
				};

			private:
				//
				template<asset::Asset AssetType>
				struct key_t
				{
					core::smart_refctd_ptr<const AssetType> asset = {};
					size_t uniqueCopyGroupID = 0;
					patch_t<AssetType> patch = {};
				};
				template<asset::Asset AssetType>
				struct HashEquals
				{
					using is_transparent = void;

					inline size_t operator()(const key_t<AssetType>& key) const
					{
						return operator()(lookup_t<AssetType>{key.asset.get(),key.uniqueCopyGroupID,&key.patch});
					}
					inline size_t operator()(const lookup_t<AssetType>& lookup) const
					{
						blake3_hasher hasher;
						blake3_hasher_init(&hasher);
						lookup.patch->hasher_update(hasher);
						const auto longHash = core::blake3_hasher_finalize(hasher);
						return std::hash<core::blake3_hash_t>()(longHash);
					}

					inline bool operator()(const key_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return lhs.asset.get()==rhs.asset.get() && lhs.uniqueCopyGroupID==rhs.uniqueCopyGroupID && lhs.patch==rhs.patch;
					}
					inline bool operator()(const key_t<AssetType>& lhs, const lookup_t<AssetType>& rhs) const
					{
						return lhs.asset.get()==rhs.asset && lhs.uniqueCopyGroupID==rhs.uniqueCopyGroupID && rhs.patch && lhs.patch==*rhs.patch;
					}
					inline bool operator()(const lookup_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return lhs.asset==rhs.asset.get() && lhs.uniqueCopyGroupID==rhs.uniqueCopyGroupID && lhs.patch && *lhs.patch==rhs.patch;
					}
				};
				template<asset::Asset AssetType>
				using container_t = core::unordered_map<key_t<AssetType>,core::blake3_hash_t,HashEquals<AssetType>,HashEquals<AssetType>>;

			public:
				//
				template<asset::Asset AssetType>
				inline core::blake3_hash_t hash(const lookup_t<AssetType>& toHash)
				{
					assert(toHash.valid());
					// consult cache
					auto& container = std::get<container_t<AssetType>>(m_containers);
					auto foundIt = container.find<lookup_t<AssetType>>(toHash);
					const bool found = foundIt!=container.end();
					// if found and we trust then return the cached hash
					if (toHash.hashTrustLevel==0 && found)
						return foundIt->second;
					// proceed with full hash computation
					core::blake3_hash_t retval;
					{
						blake3_hasher hasher;
						blake3_hasher_init(&hasher);
						// We purposefully don't hash asset pointer, we hash the contents instead
						//core::blake3_hasher_update(hasher,toHash.asset);
						core::blake3_hasher_update(hasher,toHash.uniqueCopyGroupID);
						const auto trustLevel = toHash.hashTrustLevel ? (toHash.hashTrustLevel-1):0;
//TODO					hash_impl(&hasher,toHash.asset,toHash.patch,trustLevel);
						retval = core::blake3_hasher_finalize(hasher);
					}
					if (found) // replace stale entry
						foundIt->second = retval;
					else // insert new entry
					{
						auto insertIt = container.insert(foundIt,{
							{
								.asset = core::smart_refctd_ptr<const AssetType>(toHash.asset),
								.uniqueCopyGroupID = toHash.uniqueCopyGroupID,
								.patch = *toHash.patch
							},
						retval});
						assert(HashEquals<AssetType>()(insertIt->first,toHash) && insertIt->second==retval);
					}
					return retval;
				}
				// The `hashTrust` level gets ignored (TODO: shall we use it to recurse?)
				template<asset::Asset AssetType>
				inline bool erase(const lookup_t<AssetType>& what)
				{
					return std::get<container_t<AssetType>>(m_containers).erase(what)>0;
				}
				// Warning: Linear Search! Super slow!
				template<asset::Asset AssetType>
				inline bool erase(const AssetType* asset)
				{
					return core::erase_if(std::get<container_t<AssetType>>(m_containers),[asset](const auto& entry)->bool
						{
							auto const& [key,value] = entry;
							return key.asset==asset;
						}
					);
				}
				// An asset being pointed to can mutate and that would invalidate the hash, this recomputes all hashes.
				NBL_API void eraseStale();
				// Clear the cache for a given type
				template<asset::Asset AssetType>
				inline void clear()
				{
					std::get<container_t<AssetType>>(m_containers).clear();
				}
				// Clear the caches completely
				inline void clear()
				{
					core::for_each_in_tuple(m_containers,[](auto& container)->void{container.clear();});
				}

			private:
				//
				core::tuple_transform_t<container_t,supported_asset_types> m_containers;
		};
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
				inline const auto find(const core::blake3_hash_t& hash) const
				{
					const auto end = m_forwardMap.end();
					const auto found = m_forwardMap.find(hash);
					if (found!=end)
						return found->second.get();
					return end;
				}
				inline const auto find(const asset_cached_t<AssetType>::type& gpuObject) const
				{
					const auto end = m_reverseMap.end();
					const auto found = m_reverseMap.find(gpuObject);
					if (found!=end)
						return found->second;
				}

				inline void merge(const CCache<AssetType>& other)
				{
					m_forwardMap.insert(other.m_forwardMap.begin(),other.m_forwardMap.end());
					m_reverseMap.insert(other.m_reverseMap.begin(),other.m_reverseMap.end());
				}

            private:
				// The blake3 hash is quite fat (256bit), so we don't actually store a full asset ref for comparison.
				// Assuming a uniform distribution of keys and perfect hashing, we'd expect a collision on average every 2^256 asset loads.
				// Or if you actually calculate the P(X>1) for any reasonable number of asset loads (k trials), the Poisson CDF will be pratically 0.
				core::unordered_map<core::blake3_hash_t,asset_cached_t<AssetType>> m_forwardMap;
				core::unordered_map<asset_cached_t<AssetType>,core::blake3_hash_t> m_reverseMap;
        };

		// A meta class to encompass all the Assets you might want to convert at once
        struct SInputs
        {
			struct instance_t
			{
				inline bool operator==(const instance_t&) const = default;

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

			// recommended you set this
			system::logger_opt_ptr logger = nullptr;

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
				using vector_t = core::vector<asset_cached_t<AssetType>>;

				// until `convert` is called, this will only contain valid entries for items already found in `SInput::readCache`
				template<asset::Asset AssetType>
				std::span<const asset_cached_t<AssetType>> getGPUObjects() const {return std::get<vector_t<AssetType>>(m_gpuObjects);}

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
			virtual inline SIntendedSubmitInfo& getFinalOwnerSubmit(const IDeviceMemoryBacked* imageOrBuffer, const core::blake3_hash_t& createdFrom)
			{
				return compute;
			}
			// By default we always insert into the cache
			virtual inline bool writeCache(const core::blake3_hash_t& createdFrom)
			{
				return true;
			}

			// submits the buffered up cals 
			NBL_API ISemaphore::future_t<bool> autoSubmit();

			// recommended you set this
			system::logger_opt_ptr logger = nullptr;
			// TODO: documentation
			SIntendedSubmitInfo transfer = {};
			SIntendedSubmitInfo compute = {};
			// required for Buffer or Image upload operations
			IUtilities* utilities = nullptr;
			// optional, useful for shaders
			asset::IShaderCompiler::CCache* readShaderCache = nullptr;
			asset::IShaderCompiler::CCache* writeShaderCache = nullptr;
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