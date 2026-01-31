#ifndef _NBL_EXAMPLES_C_GEOMETRY_CREATOR_SCENE_H_INCLUDED_
#define _NBL_EXAMPLES_C_GEOMETRY_CREATOR_SCENE_H_INCLUDED_


#include <nabla.h>
#include <type_traits>
#include "nbl/asset/utils/CGeometryCreator.h"

namespace nbl::examples
{

class CGeometryCreatorScene : public core::IReferenceCounted
{
#define EXPOSE_NABLA_NAMESPACES \
			using namespace nbl::core; \
			using namespace nbl::system; \
			using namespace nbl::asset; \
			using namespace nbl::video
	public:

		struct SGeometryEntry
		{
			std::string name;
			core::smart_refctd_ptr<const asset::ICPUPolygonGeometry> geometry;
		};
		
		struct SCreateParams
		{
			video::IQueue* transferQueue;
			video::IUtilities* utilities;
			system::ILogger* logger;
			std::span<const uint32_t> addtionalBufferOwnershipFamilies = {};
		};

		// Creates and initializes a scene. Override addGeometries() to supply custom meshes.
		template<typename SceneT = CGeometryCreatorScene, typename... Args>
		static inline core::smart_refctd_ptr<SceneT> create(SCreateParams&& params, const video::CAssetConverter::patch_t<asset::ICPUPolygonGeometry>& geometryPatch, Args&&... args)
		{
			static_assert(std::is_base_of_v<CGeometryCreatorScene, SceneT>);
			auto scene = core::smart_refctd_ptr<SceneT>(new SceneT(std::forward<Args>(args)...), core::dont_grab);
			if (!scene->initialize(std::move(params), geometryPatch))
				return nullptr;
			return scene;
		}

		//
		struct SInitParams
		{
			core::vector<core::smart_refctd_ptr<const video::IGPUPolygonGeometry>> geometries;
			core::vector<std::string> geometryNames;
		};
		const SInitParams& getInitParams() const {return m_init;}

	protected:
		inline CGeometryCreatorScene() = default;

		// Override to supply custom geometries, names are used as UI labels
		virtual core::vector<SGeometryEntry> addGeometries(asset::CGeometryCreator* creator) const
		{
			core::vector<SGeometryEntry> entries;
			entries.push_back({ "Cube", creator->createCube({ 1.f,1.f,1.f }) });
			entries.push_back({ "Rectangle", creator->createRectangle({ 1.5f,3.f }) });
			entries.push_back({ "Disk", creator->createDisk(2.f, 30) });
			entries.push_back({ "Sphere", creator->createSphere(2, 16, 16) });
			entries.push_back({ "Cylinder", creator->createCylinder(2, 2, 20) });
			entries.push_back({ "Cone", creator->createCone(2, 3, 10) });
			entries.push_back({ "Icosphere", creator->createIcoSphere(1, 4, true) });
			entries.push_back({ "Grid", creator->createGrid({ 32u, 32u }) });
			return entries;
		}

		inline bool initialize(SCreateParams&& params, const video::CAssetConverter::patch_t<asset::ICPUPolygonGeometry>& geometryPatch)
		{
			EXPOSE_NABLA_NAMESPACES;
			auto* logger = params.logger;
			assert(logger);
			if (!params.transferQueue)
			{
				logger->log("Pass a non-null `IQueue* transferQueue`!",ILogger::ELL_ERROR);
				return false;
			}
			if (!params.utilities)
			{
				logger->log("Pass a non-null `IUtilities* utilities`!",ILogger::ELL_ERROR);
				return false;
			}

			SInitParams init = {};
			core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
			// create out geometries
			{
				auto creator = core::make_smart_refctd_ptr<CGeometryCreator>();
				auto entries = addGeometries(creator.get());
				if (entries.empty())
					return false;

				init.geometryNames.reserve(entries.size());
				geometries.reserve(entries.size());
				for (auto& entry : entries)
				{
					if (!entry.geometry)
						continue;
					init.geometryNames.emplace_back(entry.name);
					geometries.push_back(std::move(entry.geometry));
				}

				if (geometries.empty())
					return false;
			}
			init.geometries.reserve(init.geometryNames.size());

			// convert the geometries
			{
				auto device = params.utilities->getLogicalDevice();
				smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({.device=device});

				const auto transferFamily = params.transferQueue->getFamilyIndex();

				struct SInputs : CAssetConverter::SInputs
				{
					virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBuffer* buffer, const CAssetConverter::patch_t<asset::ICPUBuffer>& patch) const
					{
						return sharedBufferOwnership;
					}

					core::vector<uint32_t> sharedBufferOwnership;
				} inputs = {};
				core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(),geometryPatch);
				{
					inputs.logger = logger;
					std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = {&geometries.front().get(),geometries.size()};
					std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
					// set up shared ownership so we don't have to 
					core::unordered_set<uint32_t> families;
					families.insert(transferFamily);
					families.insert(params.addtionalBufferOwnershipFamilies.begin(),params.addtionalBufferOwnershipFamilies.end());
					if (families.size()>1)
					for (const auto fam : families)
						inputs.sharedBufferOwnership.push_back(fam);
				}
				
				// reserve
				auto reservation = converter->reserve(inputs);
				if (!reservation)
				{
					logger->log("Failed to reserve GPU objects for CPU->GPU conversion!",ILogger::ELL_ERROR);
					return false;
				}

				// convert
				{
					auto semaphore = device->createSemaphore(0u);

					constexpr auto MultiBuffering = 2;
					std::array<smart_refctd_ptr<IGPUCommandBuffer>,MultiBuffering> commandBuffers = {};
					{
						auto pool = device->createCommandPool(transferFamily,IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT|IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
						pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,commandBuffers,smart_refctd_ptr<ILogger>(logger));
					}
					commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

					std::array<IQueue::SSubmitInfo::SCommandBufferInfo,MultiBuffering> commandBufferSubmits;
					for (auto i=0; i<MultiBuffering; i++)
						commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

					SIntendedSubmitInfo transfer = {};
					transfer.queue = params.transferQueue;
					transfer.scratchCommandBuffers = commandBufferSubmits;
					transfer.scratchSemaphore = {
						.semaphore = semaphore.get(),
						.value = 0u,
						.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
					};

					CAssetConverter::SConvertParams cpar = {};
					cpar.utilities = params.utilities;
					cpar.transfer = &transfer;

					// basically it records all data uploads and submits them right away
					auto future = reservation.convert(cpar);
					if (future.copy()!=IQueue::RESULT::SUCCESS)
					{
						logger->log("Failed to await submission feature!", ILogger::ELL_ERROR);
						return false;
					}
				}

				// assign outputs
				{
					auto inIt = reservation.getGPUObjects<ICPUPolygonGeometry>().data();
					for (auto outIt=init.geometryNames.begin(); outIt!=init.geometryNames.end(); inIt++)
					{
						if (inIt->value)
						{
							init.geometries.push_back(inIt->value);
							outIt++;
						}
						else
						{
							logger->log("Failed to convert ICPUPolygonGeometry %s to GPU!",ILogger::ELL_ERROR,outIt->c_str());
							outIt = init.geometryNames.erase(outIt);
						}
					}
				}
			}

			m_init = std::move(init);
			return true;
		}

		SInitParams m_init;
#undef EXPOSE_NABLA_NAMESPACES
};

}
#endif
