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
		// Shader, DSLayout, PipelineLayout, Compute Pipeline
		// Renderpass, Graphics Pipeline
		// Buffer, BufferView, Sampler, Image, Image View, Bottom Level AS, Top Level AS, Descriptor Set, Framebuffer  
		// Buffer -> SRange, patched usage, owner(s)
		// BufferView -> SRange, promoted format
		// Sampler -> Clamped Params (only aniso, really)
		// Image -> this, patched usage, promoted format
		// Image View -> ref to patched Image, patched usage, promoted format
		// Descriptor Set -> unique layout, 
		using supported_asset_types = core::type_list<
			asset::ICPUSampler,
			asset::ICPUShader,
			asset::ICPUBuffer,
			// acceleration structures
			// image,
//			asset::ICPUBufferView,
			// image view
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout,
			asset::ICPUPipelineCache,
			asset::ICPUComputePipeline
			// framebuffer, renderpass and graphics pipeline
			// descriptor sets
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
		struct patch_impl_t<asset::ICPUShader>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUShader);

				inline bool valid() const {return hlsl::bitCount(static_cast<std::underlying_type_t<IGPUShader::E_SHADER_STAGE>>(stage))!=1;}

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
		struct patch_t final : patch_impl_t<AssetType>
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
		};
#define NBL_API
#if 0
		//
		template<asset::Asset AssetType>
		struct patch_callback_t
		{
			static inline patch_t<AssetType> operator()(const IAsset* user, const AssetType* dependant, const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits)
			{
				if (underlying)
					return underlying();
				else
					return patch_t<AssetType>(dependant,features,limits);
			}

			std::function<patch_t<AssetType>(const IAsset*, const AssetType*)> underlying = {};
		};
#endif
		// A class to accelerate our hash computations
		class CHashCache final : public core::IReferenceCounted
		{
			public:
				//
				template<asset::Asset AssetType>
				struct lookup_t
				{
					inline bool valid() const
					{
						//return asset && features && limits && patch && patch->valid();
						return asset && patch && patch->valid();
					}

					const AssetType* asset = nullptr;
					const patch_t<AssetType>* patch = {};
				};

			private:
				//
				template<asset::Asset AssetType>
				struct key_t
				{
					core::smart_refctd_ptr<const AssetType> asset = {};
					patch_t<AssetType> patch = {};
				};
				template<asset::Asset AssetType>
				struct HashEquals
				{
					using is_transparent = void;

					inline size_t operator()(const key_t<AssetType>& key) const
					{
						return operator()(lookup_t<AssetType>{key.asset.get(),&key.patch});
					}
					inline size_t operator()(const lookup_t<AssetType>& lookup) const
					{
						core::blake3_hasher hasher;
						hasher << ptrdiff_t(lookup.asset);
						hasher << *lookup.patch;
						// put long hash inside a small hash
						return std::hash<core::blake3_hash_t>()(static_cast<core::blake3_hash_t>(hasher));
					}

					inline bool operator()(const key_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return lhs.asset.get()==rhs.asset.get() && lhs.patch==rhs.patch;
					}
					inline bool operator()(const key_t<AssetType>& lhs, const lookup_t<AssetType>& rhs) const
					{
						return lhs.asset.get()==rhs.asset && rhs.patch && lhs.patch==*rhs.patch;
					}
					inline bool operator()(const lookup_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return lhs.asset==rhs.asset.get() && lhs.patch && *lhs.patch==rhs.patch;
					}
				};
				template<asset::Asset AssetType>
				using container_t = core::unordered_map<key_t<AssetType>,core::blake3_hash_t,HashEquals<AssetType>,HashEquals<AssetType>>;

			public:
				inline CHashCache() = default;

				//
				template<asset::Asset AssetType>
				inline container_t<AssetType>::iterator find(const lookup_t<AssetType>& assetAndPatch)
				{
					return std::get<container_t<AssetType>>(m_containers).find<lookup_t<AssetType>>(assetAndPatch);
				}
				template<asset::Asset AssetType>
				inline container_t<AssetType>::const_iterator find(const lookup_t<AssetType>& assetAndPatch) const
				{
					return std::get<container_t<AssetType>>(m_containers).find<lookup_t<AssetType>>(assetAndPatch);
				}
				template<asset::Asset AssetType>
				inline container_t<AssetType>::const_iterator end() const
				{
					return std::get<container_t<AssetType>>(m_containers).end();
				}

				//
				template<asset::Asset AssetType>
				struct hash_request_t : lookup_t<AssetType>
				{
					// the callbacks are only called on dependants to construct/retrieve correct patches
#if 0
					const SPhysicalDeviceFeatures* features = nullptr;
					const SPhysicalDeviceLimits* limits = nullptr;
					core::tuple_transform_t<patch_callback_t,supported_asset_types> patchCallbacks = {};
#endif
					// how deep from `asset` do we start trusting the cache to contain correct non stale hashes
					uint32_t cacheMistrustLevel = 0;
				};
				template<asset::Asset AssetType>
				inline core::blake3_hash_t hash(const hash_request_t<AssetType>& toHash)
				{
					assert(toHash.valid());
					// consult cache
					auto foundIt = find(toHash);
					auto& container = std::get<container_t<AssetType>>(m_containers);
					const bool found = foundIt!=container.end();
					// if found and we trust then return the cached hash
					if (toHash.cacheMistrustLevel==0 && found)
						return foundIt->second;
					// proceed with full hash computation
					core::blake3_hasher hasher;
					{
						// We purposefully don't hash asset pointer, we hash the contents instead
						//core::blake3_hasher_update(hasher,toHash.asset);
						const auto nextMistrustLevel = toHash.cacheMistrustLevel ? (toHash.cacheMistrustLevel-1):0;
						hash_impl(hasher,toHash.asset,*toHash.patch,nextMistrustLevel);
					}
					// don't call finalize twice for no reason
					const auto retval = static_cast<core::blake3_hash_t>(hasher);
					if (found) // replace stale entry
						foundIt->second = retval;
					else // insert new entry
					{
						auto insertIt = container.insert(foundIt,{
							{
								.asset = core::smart_refctd_ptr<const AssetType>(toHash.asset),
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
					// TODO: improve by cycling through possible patches
					return core::erase_if(std::get<container_t<AssetType>>(m_containers),[asset](const auto& entry)->bool
						{
							auto const& [key,value] = entry;
							return key.asset==asset;
						}
					);
				}
// TODO: `eraseStale(const IAsset*)` which erases a subgraph?
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
				inline ~CHashCache() = default;

				//
				template<asset::Asset AssetType>
				void hash_impl(core::blake3_hasher& hasher, const AssetType* asset, const patch_t<AssetType>& patch, const uint32_t nextMistrustLevel);

				//
				core::tuple_transform_t<container_t,supported_asset_types> m_containers;
		};
		// Typed Cache (for a particular AssetType)
		template<asset::Asset AssetType>
        class CCache final
        {
			public:
				inline CCache() = default;
				inline CCache(const CCache&) = default;
				inline CCache(CCache&&) = default;
				inline ~CCache() = default;

				inline CCache& operator=(const CCache&) = default;
				inline CCache& operator=(CCache&&) = default;

				// Make it clear to users that we don't look up just by the asset content hash
				struct key_t
				{
					inline key_t(const core::blake3_hash_t& contentHash, const size_t uniqueCopyGroupID) : value(contentHash)
					{
						reinterpret_cast<size_t*>(value.data)[0] ^= uniqueCopyGroupID;
					}

					inline bool operator==(const key_t&) const = default;

					// The blake3 hash is quite fat (256bit), so we don't actually store a full asset ref for comparison.
					// Assuming a uniform distribution of keys and perfect hashing, we'd expect a collision on average every 2^256 asset loads.
					// Or if you actually calculate the P(X>1) for any reasonable number of asset loads (k trials), the Poisson CDF will be pratically 0.
					core::blake3_hash_t value;
				};

				// no point returning iterators to inserted positions, they're not stable
				inline bool insert(const key_t& _key, const asset_cached_t<AssetType>& _gpuObj)
				{
					auto [unused0,insertedF] = m_forwardMap.emplace(_key,_gpuObj);
					if (!insertedF)
						return false;
					auto [unused1,insertedR] = m_reverseMap.emplace(_gpuObj.get(),_key);
					assert(insertedR);
					return true;
				}

				// fastest lookup
				inline const asset_cached_t<AssetType>* find(const key_t& _key) const
				{
					const auto end = m_forwardMap.end();
					const auto found = m_forwardMap.find(_key);
					if (found!=end)
						return &found->second;
					return nullptr;
				}
				inline const key_t* find(asset_traits<AssetType>::lookup_t gpuObject) const
				{
					const auto end = m_reverseMap.end();
					const auto found = m_reverseMap.find(gpuObject);
					if (found!=end)
						return &found->second;
					return nullptr;
				}

			private:
				struct ForwardHash
				{
					inline size_t operator()(const key_t& key) const
					{
						return std::hash<core::blake3_hash_t>()(key.value);
					}
				};
				core::unordered_map<key_t,asset_cached_t<AssetType>,ForwardHash> m_forwardMap;
				core::unordered_map<typename asset_traits<AssetType>::lookup_t,key_t> m_reverseMap;

			public:
				// fastest erase
				inline bool erase(decltype(m_forwardMap)::const_iterator fit, decltype(m_reverseMap)::const_iterator rit)
				{
					if (fit->first!=rit->second || fit->second!=rit->first)
						return false;
					m_reverseMap.erase(rit);
					m_forwardMap.erase(fit);
					return true;
				}
				inline bool erase(decltype(m_forwardMap)::const_iterator it)
				{
					return erase(it,find(it->second));
				}
				inline bool erase(decltype(m_reverseMap)::const_iterator it)
				{
					return erase(find(it->second),it);
				}

				inline void merge(const CCache<AssetType>& other)
				{
					m_forwardMap.insert(other.m_forwardMap.begin(),other.m_forwardMap.end());
					m_reverseMap.insert(other.m_reverseMap.begin(),other.m_reverseMap.end());
				}
        };

		// A meta class to encompass all the Assets you might want to convert at once
        struct SInputs
        {
			// Normally all references to the same IAsset* would spawn the same IBackendObject*.
			// You need to tell us if an asset needs multiple copies, separate for each user. The return value of this function dictates what copy of the asset each user gets.
			// Each unique integer value returned for a given input `dependant` "spawns" a new copy.
			// Note that the group ID is the same size as a pointer, so you can e.g. cast a pointer of the user (parent reference) to size_t and use that for a unique copy for the user.
			// Note that we also call it with `user=={nullptr,0}` for each entry in `SInputs::assets`.
			// NOTE: You might get extra copies within the same group ID due to inability to patch entries
			virtual inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const asset::IAsset* user, const asset::IAsset* dependant) const
			{
				return 0;
			}

			// Typed Range of Inputs of the same type
            template<asset::Asset AssetType>
            using asset_span_t = std::span<const typename asset_traits<AssetType>::asset_t* const>;
            template<asset::Asset AssetType>
            using patch_span_t = std::span<const patch_t<AssetType>>;

			// can be `nullptr` and even equal to `this`
			const CAssetConverter* readCache = nullptr;

			// recommended you set this
			system::logger_opt_ptr logger = nullptr;

			// A type-sorted non-polymorphic list of "root assets"
			core::tuple_transform_t<asset_span_t,supported_asset_types> assets = {};
			// Optional: Whatever is not in `patches` will generate a default patch
			core::tuple_transform_t<patch_span_t,supported_asset_types> patches = {};

			// optional, useful for shaders
			asset::IShaderCompiler::CCache* readShaderCache = nullptr;
			asset::IShaderCompiler::CCache* writeShaderCache = nullptr;
			IGPUPipelineCache* pipelineCache = nullptr;

			// Leave this as `nullptr` unless you know what you're doing. If you supply a hash cache,
			// and it has stale hashes (asset or its dependants mutated since hash computed),
			// then you can get hash mismatches which will lead to UB or conversion failures.
			CHashCache* hashCache = nullptr;
        };
        struct SResults
        {
			public:
				inline SResults(SResults&&) = default;
				inline SResults(const SResults&) = delete;
				inline ~SResults() = default;
				inline SResults& operator=(const SResults&) = delete;
				inline SResults& operator=(SResults&&) = default;

				// What queues you'll need to run the submit
				inline core::bitflag<IQueue::FAMILY_FLAGS> getRequiredQueueFlags() const {return m_queueFlags;}

				// until `convert` is called, this will only contain valid entries for items already found in `SInput::readCache`
				template<asset::Asset AssetType>
				std::span<const asset_cached_t<AssetType>> getGPUObjects() const {return std::get<vector_t<AssetType>>(m_gpuObjects);}

			private:
				friend class CAssetConverter;
				friend struct SConvertParams;

				inline SResults() = default;

				// for every entry in the input array, we have this mapped 1:1
				template<asset::Asset AssetType>
				using vector_t = core::vector<asset_cached_t<AssetType>>;
				core::tuple_transform_t<vector_t,supported_asset_types> m_gpuObjects = {};
				
				// we don't insert into the writeCache until conversions are successful
				core::tuple_transform_t<CCache,supported_asset_types> m_stagingCaches;
#if 0
				//
				template<asset::Asset AssetType>
				struct reservation_t
				{
					patch_t<AssetType> patch = {};
					const AssetType* asset = nullptr;
					// index to write into `m_gpuObjects`, if past the end the reservation is for an implicit item
					size_t outputIndex = ~0x0ull;
				};
				template<asset::Asset AssetType>
				using reservation_container_t = core::unordered_map<core::blake3_hash_t,reservation_t<AssetType>>;
				core::tuple_transform_t<reservation_container_t,supported_asset_types> m_reservations;
#endif
				//
				core::bitflag<IQueue::FAMILY_FLAGS> m_queueFlags = IQueue::FAMILY_FLAGS::NONE;
        };
		// First Pass: Explore the DAG of Assets and "gather" patch infos and create equivalent GPU Objects.
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
			// TODO: documentation (and allow same queue/same intended submit)
			SIntendedSubmitInfo transfer = {};
			SIntendedSubmitInfo compute = {};
			// required for Buffer or Image upload operations
			IUtilities* utilities = nullptr;
		};
		// Second Pass: Actually fill the GPU Objects with data
		NBL_API bool convert(SResults&& reservations, SConvertParams& params);
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
		
		template<asset::Asset AssetType>
		inline CCache<AssetType>& getCache()
		{
			return std::get<CCache<AssetType>>(m_caches);
		}

        SCreationParams m_params;
		core::tuple_transform_t<CCache,supported_asset_types> m_caches;
};

}
#endif