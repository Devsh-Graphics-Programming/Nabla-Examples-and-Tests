#ifndef _NBL_EXAMPLES_C_GEOMETRY_CREATOR_SCENE_H_INCLUDED_
#define _NBL_EXAMPLES_C_GEOMETRY_CREATOR_SCENE_H_INCLUDED_


#include <nabla.h>
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
		//
		struct SCreateParams
		{
			video::IQueue* transferQueue;
			video::IUtilities* utilities;
			system::ILogger* logger;
			std::span<const uint32_t> addtionalBufferOwnershipFamilies = {};
		};
		static inline core::smart_refctd_ptr<CGeometryCreatorScene> create(SCreateParams&& params, const video::CAssetConverter::patch_t<asset::ICPUPolygonGeometry>& geometryPatch)
		{
			EXPOSE_NABLA_NAMESPACES;
			auto* logger = params.logger;
			assert(logger);
			if (!params.transferQueue)
			{
				logger->log("Pass a non-null `IQueue* transferQueue`!",ILogger::ELL_ERROR);
				return nullptr;
			}
			if (!params.utilities)
			{
				logger->log("Pass a non-null `IUtilities* utilities`!",ILogger::ELL_ERROR);
				return nullptr;
			}


			SInitParams init = {};
			core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
			// create out geometries
			{
				auto addGeometry = [&init,&geometries](const std::string_view name, smart_refctd_ptr<const ICPUPolygonGeometry>&& geom)->void
				{
					init.geometryNames.emplace_back(name);
					geometries.push_back(std::move(geom));
				};

				auto creator = core::make_smart_refctd_ptr<CGeometryCreator>();
				/* TODO: others
				ReferenceObjectCpu {.meta = {.type = OT_CUBE, .name = "Cube Mesh" }, .shadersType = GP_BASIC, .data = gc->createCubeMesh(nbl::core::vector3df(1.f, 1.f, 1.f)) },
				ReferenceObjectCpu {.meta = {.type = OT_SPHERE, .name = "Sphere Mesh" }, .shadersType = GP_BASIC, .data = gc->createSphereMesh(2, 16, 16) },
				ReferenceObjectCpu {.meta = {.type = OT_CYLINDER, .name = "Cylinder Mesh" }, .shadersType = GP_BASIC, .data = gc->createCylinderMesh(2, 2, 20) },
				ReferenceObjectCpu {.meta = {.type = OT_RECTANGLE, .name = "Rectangle Mesh" }, .shadersType = GP_BASIC, .data = gc->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3)) },
				ReferenceObjectCpu {.meta = {.type = OT_DISK, .name = "Disk Mesh" }, .shadersType = GP_BASIC, .data = gc->createDiskMesh(2, 30) },
				ReferenceObjectCpu {.meta = {.type = OT_ARROW, .name = "Arrow Mesh" }, .shadersType = GP_BASIC, .data = gc->createArrowMesh() },
				ReferenceObjectCpu {.meta = {.type = OT_CONE, .name = "Cone Mesh" }, .shadersType = GP_CONE, .data = gc->createConeMesh(2, 3, 10) },
				ReferenceObjectCpu {.meta = {.type = OT_ICOSPHERE, .name = "Icoshpere Mesh" }, .shadersType = GP_ICO, .data = gc->createIcoSphere(1, 3, true) }
				*/
				addGeometry("Cube",creator->createCube({1.f,1.f,1.f}));
				addGeometry("Rectangle",creator->createRectangle({1.5f,3.f}));
				addGeometry("Disk",creator->createDisk(2.f,30));
				addGeometry("Sphere", creator->createSphere(2, 16, 16));
				addGeometry("Cylinder", creator->createCylinder(2, 2, 20));
				addGeometry("Cone", creator->createCone(2, 3, 10));
				addGeometry("Icosphere", creator->createIcoSphere(1, 4, true));
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
					return nullptr;
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
						return nullptr;
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

			return smart_refctd_ptr<CGeometryCreatorScene>(new CGeometryCreatorScene(std::move(init)),dont_grab);
		}

		//
		struct SInitParams
		{
			core::vector<core::smart_refctd_ptr<const video::IGPUPolygonGeometry>> geometries;
			core::vector<std::string> geometryNames;
		};
		const SInitParams& getInitParams() const {return m_init;}

	protected:
		inline CGeometryCreatorScene(SInitParams&& _init) : m_init(std::move(_init)) {}

		SInitParams m_init;
#undef EXPOSE_NABLA_NAMESPACES
};

}
#endif