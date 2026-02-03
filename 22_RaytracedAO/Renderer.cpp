#include <numeric>
#include <filesystem>

#include "Renderer.h"

#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/asset/filters/CFillImageFilter.h"
#include "../source/Nabla/COpenCLHandler.h"
#include "COpenGLDriver.h"


#ifndef _NBL_BUILD_OPTIX_
	#define __C_CUDA_HANDLER_H__ // don't want CUDA declarations and defines to pollute here
#endif

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;


constexpr uint32_t kOptiXPixelSize = sizeof(uint16_t)*3u;

// TODO: make these util function in `IDescriptorSetLayout` -> Assign: @Vib
auto fillIotaDescriptorBindingDeclarations = [](auto* outBindings, uint32_t accessFlags, uint32_t count, asset::E_DESCRIPTOR_TYPE descType=asset::EDT_INVALID, uint32_t startIndex=0u) -> void
{
	for (auto i=0u; i<count; i++)
	{
		outBindings[i].binding = i+startIndex;
		outBindings[i].type = descType;
		outBindings[i].count = 1u;
		outBindings[i].stageFlags = static_cast<ISpecializedShader::E_SHADER_STAGE>(accessFlags);
		outBindings[i].samplers = nullptr;
	}
};

Renderer::Renderer(IVideoDriver* _driver, IAssetManager* _assetManager, scene::ISceneManager* _smgr, bool deferDenoise, bool useDenoiser) :
		m_useDenoiser(useDenoiser), m_deferDenoise(deferDenoise), m_driver(_driver), m_smgr(_smgr), m_assetManager(_assetManager),
		m_rrManager(ext::RadeonRays::Manager::create(m_driver)),
		m_prevView(), m_prevCamTform(), m_sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX), m_maxAreaLightLuma(0.f),
		m_framesDispatched(0u), m_rcpPixelSize{0.f,0.f},
		m_staticViewData{ {0u,0u},0u,0u,0u,0u,false,core::infinity<float>(),{}}, m_raytraceCommonData{0.f,0u,0u,0u,core::matrix3x4SIMD()},
		m_indirectDrawBuffers{nullptr},m_cullPushConstants{core::matrix4SIMD(),1.f,0u,0u,0u},m_cullWorkGroups(0u),
		m_raygenWorkGroups{0u,0u},m_colorBuffer(nullptr),
		m_envMapImportanceSampling(_driver)
{
	// TODO: reimplement
	m_useDenoiser = false;

	// set up raycount buffers
	{
		const uint32_t zeros[RAYCOUNT_N_BUFFERING] = { 0u };
		m_rayCountBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t)*RAYCOUNT_N_BUFFERING,zeros);
		IDriverMemoryBacked::SDriverMemoryRequirements reqs;
		reqs.vulkanReqs.size = sizeof(uint32_t);
		reqs.vulkanReqs.alignment = alignof(uint32_t);
		reqs.vulkanReqs.memoryTypeBits = ~0u;
		reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
		reqs.mappingCapability = IDriverMemoryAllocation::EMCF_COHERENT|IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ;
		reqs.prefersDedicatedAllocation = 0u;
		reqs.requiresDedicatedAllocation = 0u;
		m_littleDownloadBuffer = m_driver->createGPUBufferOnDedMem(reqs);
		m_littleDownloadBuffer->getBoundMemory()->mapMemoryRange(IDriverMemoryAllocation::EMCAF_READ,{0,sizeof(uint32_t)});
	}

	// no deferral for now
	m_fragGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../fillVisBuffer.frag");

	// set up Visibility Buffer pipeline
	{
		IGPUDescriptorSetLayout::SBinding binding;
		fillIotaDescriptorBindingDeclarations(&binding,ISpecializedShader::ESS_VERTEX|ISpecializedShader::ESS_FRAGMENT,1u,asset::EDT_STORAGE_BUFFER);

		m_rasterInstanceDataDSLayout = m_driver->createGPUDescriptorSetLayout(&binding,&binding+1u);
	}
	{
		constexpr auto additionalGlobalDescriptorCount = 5u;
		IGPUDescriptorSetLayout::SBinding bindings[additionalGlobalDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE|ISpecializedShader::ESS_VERTEX|ISpecializedShader::ESS_FRAGMENT,additionalGlobalDescriptorCount,asset::EDT_STORAGE_BUFFER);

		m_additionalGlobalDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+additionalGlobalDescriptorCount);
	}
	{
		constexpr auto cullingDescriptorCount = 3u;
		IGPUDescriptorSetLayout::SBinding bindings[cullingDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE|ISpecializedShader::ESS_VERTEX,cullingDescriptorCount,asset::EDT_STORAGE_BUFFER);
		bindings[2u].count = 2u;

		m_cullDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+cullingDescriptorCount);
	}
	m_perCameraRasterDSLayout = core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(m_cullDSLayout);
	
	{
		constexpr auto raytracingCommonDescriptorCount = 11u;
		IGPUDescriptorSetLayout::SBinding bindings[raytracingCommonDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,raytracingCommonDescriptorCount);
		bindings[0].type = asset::EDT_UNIFORM_BUFFER;
		bindings[1].type = asset::EDT_UNIFORM_TEXEL_BUFFER;
		bindings[2].type = asset::EDT_STORAGE_IMAGE;
		bindings[3].type = asset::EDT_STORAGE_BUFFER;
		bindings[4].type = asset::EDT_STORAGE_BUFFER;
		bindings[5].type = asset::EDT_STORAGE_IMAGE;
		bindings[6].type = asset::EDT_STORAGE_IMAGE;
		bindings[7].type = asset::EDT_STORAGE_IMAGE;
		bindings[8].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[9].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[10].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		m_commonRaytracingDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+raytracingCommonDescriptorCount);
	}

	ISampler::SParams samplerParams;
	samplerParams.TextureWrapU = samplerParams.TextureWrapV = samplerParams.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;
	samplerParams.MinFilter = samplerParams.MaxFilter = ISampler::ETF_NEAREST;
	samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
	samplerParams.AnisotropicFilter = 0u;
	samplerParams.CompareEnable = false;
	auto sampler = m_driver->createGPUSampler(samplerParams);
	{
		constexpr auto raygenDescriptorCount = 3u;
		IGPUDescriptorSetLayout::SBinding bindings[raygenDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,raygenDescriptorCount,EDT_COMBINED_IMAGE_SAMPLER);
		bindings[0].samplers = &sampler;
		bindings[1].samplers = &sampler;
		bindings[2].type = asset::EDT_STORAGE_IMAGE;

		m_raygenDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+raygenDescriptorCount);
	}
	{
		constexpr auto closestHitDescriptorCount = 2u;
		IGPUDescriptorSetLayout::SBinding bindings[2];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,closestHitDescriptorCount,EDT_STORAGE_BUFFER);

		m_closestHitDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+closestHitDescriptorCount);
	}
	{
		constexpr auto resolveDescriptorCount = 8u;
		IGPUDescriptorSetLayout::SBinding bindings[resolveDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,resolveDescriptorCount);
		bindings[0].type = asset::EDT_UNIFORM_BUFFER;
		bindings[1].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[1].samplers = &sampler;
		bindings[2].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[2].samplers = &sampler;
		bindings[3].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[3].samplers = &sampler;
		bindings[4].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[4].samplers = &sampler;
		bindings[5].type = asset::EDT_STORAGE_IMAGE;
		bindings[6].type = asset::EDT_STORAGE_IMAGE;
		bindings[7].type = asset::EDT_STORAGE_IMAGE;

		m_resolveDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+resolveDescriptorCount);
	}

	if(m_deferDenoise)
		m_deferDenoiseFile.open(DEFER_DENOISE_HOOK_FILE_NAME, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
}

Renderer::~Renderer()
{
	deinitSceneResources();
	deinitRenderer();
	finalizeDeferredDenoise();
}


Renderer::InitializationData Renderer::initSceneObjects(const SAssetBundle& meshes)
{
	constexpr bool meshPackerUsesSSBO = true;
	using CPUMeshPacker = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;
	using GPUMeshPacker = CGPUMeshPackerV2<DrawElementsIndirectCommand_t>;

	// get primary (texture and material) global DS
	InitializationData retval = {};
	m_globalMeta  = meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>();
	assert(m_globalMeta);

	//
	{
		// extract integrator parameters
		std::stack<const ext::MitsubaLoader::CElementIntegrator*> integratorStack;
		integratorStack.push(&m_globalMeta->m_global.m_integrator);
		while (!integratorStack.empty())
		{
			auto integrator = integratorStack.top();
			integratorStack.pop();
			using Enum = ext::MitsubaLoader::CElementIntegrator::Type;
			switch (integrator->type)
			{
				case Enum::DIRECT:
					maxPathDepth = 2u;
					hideEnvironment = integrator->direct.hideEnvironment;
					break;
				case Enum::PATH:
					hideEnvironment = integrator->path.hideEnvironment;
					[[fallthrough]];
				case Enum::VOL_PATH_SIMPLE:
				case Enum::VOL_PATH:
				case Enum::BDPT:
					maxPathDepth = integrator->bdpt.maxPathDepth;
					noRussianRouletteDepth = integrator->bdpt.russianRouletteDepth-1u;
					break;
				case Enum::ADAPTIVE:
					for (size_t i=0u; i<integrator->multichannel.childCount; i++)
						integratorStack.push(integrator->multichannel.children[i]);
					break;
				case Enum::IRR_CACHE:
					assert(false);
					break;
				case Enum::MULTI_CHANNEL:
					for (size_t i=0u; i<integrator->multichannel.childCount; i++)
						integratorStack.push(integrator->multichannel.children[i]);
					break;
				default:
					break;
			};
		}

		//
		for (const auto& sensor : m_globalMeta->m_global.m_sensors)
		{
			if (maxSensorSamples<sensor.sampler.sampleCount)
				maxSensorSamples = sensor.sampler.sampleCount;
		}
	}

	//
	auto* _globalBackendDataDS = m_globalMeta->m_global.m_ds0.get();

	constexpr auto kInstanceDataDescriptorBinding = 5u;
	auto* instanceDataDescPtr = _globalBackendDataDS->getDescriptors(kInstanceDataDescriptorBinding).begin();
	assert(instanceDataDescPtr->desc->getTypeCategory()==IDescriptor::EC_BUFFER);
	auto* origInstanceData = reinterpret_cast<const ext::MitsubaLoader::instance_data_t*>(static_cast<ICPUBuffer*>(instanceDataDescPtr->desc.get())->getPointer());

	IGPUDescriptorSet::SDescriptorInfo infos[4];
	auto recordInfoBuffer = [](IGPUDescriptorSet::SDescriptorInfo& info, core::smart_refctd_ptr<IGPUBuffer>&& buf) -> void
	{
		info.buffer.size = buf->getSize();
		info.buffer.offset = 0u;
		info.desc = std::move(buf);
	};
	constexpr uint32_t writeBound = 3u;
	IGPUDescriptorSet::SWriteDescriptorSet writes[writeBound];
	auto recordSSBOWrite = [](IGPUDescriptorSet::SWriteDescriptorSet& write, IGPUDescriptorSet::SDescriptorInfo* infos, uint32_t binding, uint32_t count=1u) -> void
	{
		write.binding = binding;
		write.arrayElement = 0u;
		write.count = count;
		write.descriptorType = EDT_STORAGE_BUFFER;
		write.info = infos;
	};
	auto setDstSetOnAllWrites = [&writes,writeBound](IGPUDescriptorSet* dstSet) -> void
	{
		for (auto i=0u; i<writeBound; i++)
			writes[i].dstSet = dstSet;
	};
	// make secondary (geometry) DS
	m_additionalGlobalDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_additionalGlobalDSLayout));

	// one cull data per instace of a batch
	core::vector<CullData_t> cullData;
	{
		auto* rr = m_rrManager->getRadeonRaysAPI();
		// set up batches/meshlets, lights and culling data
		{
			auto contents = meshes.getContents();
			const auto meshCount = contents.size();
			printf("[INFO] Mesh Count: %d\n",meshCount);

			core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd;
			// split into packed batches
			{
				// one instance data per instance of a batch
				core::smart_refctd_ptr<ICPUBuffer> newInstanceDataBuffer;

				constexpr uint16_t minTrisBatch = MAX_TRIANGLES_IN_BATCH>>3u; // allow small allocations to fight fragmentation
				constexpr uint16_t maxTrisBatch = MAX_TRIANGLES_IN_BATCH;
				constexpr uint8_t minVertexSize = 
					asset::getTexelOrBlockBytesize<asset::EF_R32G32B32_SFLOAT>()+
					asset::getTexelOrBlockBytesize<asset::EF_A2R10G10B10_SNORM_PACK32>()+
					asset::getTexelOrBlockBytesize<asset::EF_R32G32_SFLOAT>();

				constexpr uint8_t kIndicesPerTriangle = 3u;
				constexpr uint16_t minIndicesBatch = minTrisBatch*kIndicesPerTriangle;

				CPUMeshPacker::AllocationParams allocParams;
				allocParams.vertexBuffSupportedByteSize = (1u<<31u)-1; // RTX cards
				allocParams.vertexBufferMinAllocByteSize = minTrisBatch*minVertexSize; // under max vertex reuse
				allocParams.indexBuffSupportedCnt = (allocParams.vertexBuffSupportedByteSize/allocParams.vertexBufferMinAllocByteSize)*minIndicesBatch;
				allocParams.indexBufferMinAllocCnt = minIndicesBatch;
				allocParams.MDIDataBuffSupportedCnt = allocParams.indexBuffSupportedCnt/minIndicesBatch;
				allocParams.MDIDataBuffMinAllocCnt = 1u; //so structs from different meshbuffers are adjacent in memory


				// TODO: after position moves to RGB21, need to split up normal from UV
				constexpr auto combinedNormalUVAttributeIx = 1;
				constexpr auto newEnabledAttributeMask = (0x1u<<combinedNormalUVAttributeIx)|0b1;

				IMeshPackerV2Base::SupportedFormatsContainer formats;
				formats.insert(EF_R32G32B32_SFLOAT);
				formats.insert(EF_R32G32B32_UINT);
				auto cpump = core::make_smart_refctd_ptr<CCPUMeshPackerV2<>>(allocParams,formats,minTrisBatch,maxTrisBatch);
				uint32_t mdiBoundMax=0u,batchInstanceBoundTotal=0u;
				core::vector<CPUMeshPacker::ReservedAllocationMeshBuffers> allocData;
				// virtually allocate and size the storage
				{
					core::vector<const ICPUMeshBuffer*> meshBuffersToProcess;
					meshBuffersToProcess.reserve(contents.size());
					// TODO: Optimize! Check which triangles need normals, bin into two separate meshbuffers, dont have normals for meshbuffers where all(abs(transpose(normals)*cross(pos1-pos0,pos2-pos0))~=1.f) 
					// TODO: Optimize! Check which materials use any textures, if meshbuffer doens't use any textures, its pipeline doesn't need UV coordinates
					// TODO: separate pipeline for stuff without UVs and separate out the barycentric derivative FBO attachment 
					// stats 
					uint32_t totalInstanceCount = 0;
					size_t totalInstancedTriangleCount = 0u;
					for (const auto& asset : contents)
					{
						auto cpumesh = static_cast<asset::ICPUMesh*>(asset.get());
						auto meshBuffers = cpumesh->getMeshBuffers();

						assert(!meshBuffers.empty());
						const uint32_t instanceCount = (*meshBuffers.begin())->getInstanceCount();
						totalInstanceCount += instanceCount;
						for (auto mbIt=meshBuffers.begin(); mbIt!=meshBuffers.end(); mbIt++)
						{
							auto meshBuffer = *mbIt;
							totalInstancedTriangleCount += (meshBuffer->getIndexCount()/3)*instanceCount;
							assert(meshBuffer->getInstanceCount()==instanceCount);
							// We'll disable certain attributes to ensure we only copy position, normal and uv attribute
							SVertexInputParams& vertexInput = meshBuffer->getPipeline()->getVertexInputParams();
							// but we'll pack normals and UVs together to save one SSBO binding, but no quantization of UVs to keep accurate floating point precision for baricentrics
							constexpr auto freeBinding = 15u;
							vertexInput.attributes[combinedNormalUVAttributeIx].binding = freeBinding;
							vertexInput.attributes[combinedNormalUVAttributeIx].format = EF_R32G32B32_UINT;
							vertexInput.attributes[combinedNormalUVAttributeIx].relativeOffset = 0u;
							vertexInput.enabledBindingFlags |= 0x1u<<freeBinding;
							vertexInput.bindings[freeBinding].inputRate = EVIR_PER_VERTEX;
							vertexInput.bindings[freeBinding].stride = 0u;
							const auto approxVxCount = IMeshManipulator::upperBoundVertexID(meshBuffer)+meshBuffer->getBaseVertex();
							
							struct CombinedNormalUV
							{
								uint32_t normal;
								float u, v;
							};
							static_assert(sizeof(CombinedNormalUV) == sizeof(float) * 3u);

							auto newBuff = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(CombinedNormalUV)*approxVxCount);
							auto* dst = reinterpret_cast<CombinedNormalUV*>(newBuff->getPointer())+meshBuffer->getBaseVertex();
							meshBuffer->setVertexBufferBinding({0u,newBuff},freeBinding);
							// copy and pack data
							const auto normalAttr = meshBuffer->getNormalAttributeIx();
							vertexInput.attributes[normalAttr].format = EF_R32_UINT;
							for (auto i=0u; i<approxVxCount; i++)
							{
								meshBuffer->getAttribute(&dst[i].normal,normalAttr,i);
								core::vectorSIMDf uv;
								meshBuffer->getAttribute(uv,2u,i);
								dst[i].u = uv.x;
								dst[i].v = uv.y;
							}
						}

						const uint32_t mdiBound = cpump->calcMDIStructMaxCount(meshBuffers.begin(),meshBuffers.end());
						mdiBoundMax = core::max(mdiBound,mdiBoundMax);
						batchInstanceBoundTotal += mdiBound*instanceCount;

						meshBuffersToProcess.insert(meshBuffersToProcess.end(),meshBuffers.begin(),meshBuffers.end());
					}
					if (totalInstancedTriangleCount >= ~0x0u)
						printf("[ERROR] Over 2^32-1 Triangles, WILL CRASH!\n");
					printf("[INFO] Total Instanced Triangles in the Scene: %d\n",totalInstancedTriangleCount);
					{
						// Radeon Rays uses 64 byte nodes in BVH2
						const auto bvh_size = size_t(totalInstancedTriangleCount)*64u*2u;
						printf("[INFO] BVH Size: At Least %d MB\n",bvh_size>>20u);
						if ((0x1ull<<31)<bvh_size)
							printf("[WARNING] BVH Larger than 2GB!\n");
						if ((0x1ull<<32)<bvh_size)
							printf("[WARNING] BVH Larger than 4GB! EXPECT HANGS AND CRASHES!\n");
					}
					printf("[INFO] Total Shape Instances in the Scene: %d\n", totalInstanceCount);
					for (auto meshBuffer : meshBuffersToProcess)
						const_cast<ICPUMeshBuffer*>(meshBuffer)->getPipeline()->getVertexInputParams().enabledAttribFlags = newEnabledAttributeMask;

					allocData.resize(meshBuffersToProcess.size());

					if (!cpump->alloc(allocData.data(),meshBuffersToProcess.begin(),meshBuffersToProcess.end()))
					{
						printf("[ERROR] Failed to Allocate Mesh data in SSBOs, quitting!\n");
						exit(-42);
					}
					cpump->shrinkOutputBuffersSize();
					cpump->instantiateDataStorage();

					pmbd.resize(meshBuffersToProcess.size());
					cullData.reserve(batchInstanceBoundTotal);

					newInstanceDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ext::MitsubaLoader::instance_data_t)*batchInstanceBoundTotal);
				}
				//
				{
					const auto totalIndexBytes = cpump->getIndexAllocator().get_allocated_size();
					printf("[INFO] Total Index Memory Consumed: %d MB\n",((totalIndexBytes-1)>>20)+1);
					const auto totalVertexBytes = cpump->getVertexAllocator().get_allocated_size();
					printf("[INFO] Total Vertex Memory Consumed: %d MB\n",((totalVertexBytes-1)>>20)+1);
				}
				// actually commit the physical memory, compute batches and set up instance data
				{
					auto allocDataIt = allocData.begin();
					auto pmbdIt = pmbd.begin();
					auto* indexPtr = reinterpret_cast<const uint16_t*>(cpump->getPackerDataStore().indexBuffer->getPointer());
					auto* vertexPtr = reinterpret_cast<const float*>(cpump->getPackerDataStore().vertexBuffer->getPointer());
					auto* mdiPtr = reinterpret_cast<DrawElementsIndirectCommand_t*>(cpump->getPackerDataStore().MDIDataBuffer->getPointer());
					auto* newInstanceData = reinterpret_cast<ext::MitsubaLoader::instance_data_t*>(newInstanceDataBuffer->getPointer());

					constexpr uint32_t kIndicesPerTriangle = 3u;
					core::vector<CPUMeshPacker::CombinedDataOffsetTable> cdot(mdiBoundMax);
					core::vector<core::aabbox3df> aabbs(mdiBoundMax);
					MDICall* mdiCall = nullptr;
					core::vector<int32_t> fatIndicesForRR(maxTrisBatch*kIndicesPerTriangle);
					for (const auto& asset : contents)
					{
						auto cpumesh = static_cast<asset::ICPUMesh*>(asset.get());
						const auto* meta = m_globalMeta ->getAssetSpecificMetadata(cpumesh);
						const auto& instanceData = meta->m_instances;
						const auto& instanceAuxData = meta->m_instanceAuxData;

						auto meshBuffers = cpumesh->getMeshBuffers();
						const uint32_t actualMdiCnt = cpump->commit(&*pmbdIt,cdot.data(),aabbs.data(),&*allocDataIt,meshBuffers.begin(),meshBuffers.end());
						allocDataIt += meshBuffers.size();
						if (actualMdiCnt==0u)
						{
							std::cout << "Commit failed" << std::endl;
							_NBL_DEBUG_BREAK_IF(true);
							pmbdIt += meshBuffers.size();
							continue;
						}

						const auto aabbMesh = cpumesh->getBoundingBox();
						// meshbuffers
						auto cdotIt = cdot.begin();
						auto aabbsIt = aabbs.begin();
						for (auto mb : meshBuffers)
						{
							assert(mb->getInstanceCount()==instanceData.size());
							const auto posAttrID = mb->getPositionAttributeIx();
							const auto* mbInstanceData = origInstanceData+mb->getBaseInstance();
							const bool frontFaceIsCCW = mb->getPipeline()->getRasterizationParams().frontFaceIsCCW;
							// batches/meshlets
							for (auto i=0u; i<pmbdIt->mdiParameterCount; i++)
							{
								const uint32_t drawCommandGUID = pmbdIt->mdiParameterOffset+i;
								auto& mdi = mdiPtr[drawCommandGUID];
								mdi.baseInstance = cullData.size();
								mdi.instanceCount = 0; // needs to be cleared, will be set by compute culling

								const uint32_t firstIndex = mdi.firstIndex;
								// set up BLAS
								const auto indexCount = mdi.count;
								std::copy_n(indexPtr+firstIndex,indexCount,fatIndicesForRR.data());
								rrShapes.emplace_back() = rr->CreateMesh(
									vertexPtr+cdotIt->attribInfo[posAttrID].getOffset()*sizeof(vec3)/sizeof(float),
									mdi.count, // could be improved if mesh packer returned the `usedVertices.size()` for every batch in the cdot
									asset::getTexelOrBlockBytesize<asset::EF_R32G32B32_SFLOAT>(),
									fatIndicesForRR.data(),
									sizeof(uint32_t)*kIndicesPerTriangle,nullptr, // radeon rays understands index stride differently to me
									indexCount/kIndicesPerTriangle
								);

								const auto thisShapeInstancesBeginIx = rrInstances.size();
								const auto& batchAABB = *aabbsIt;
								for (auto auxIt=instanceAuxData.begin(); auxIt!=instanceAuxData.end(); auxIt++)
								{
									const auto batchInstanceGUID = cullData.size();

									const auto instanceID = std::distance(instanceAuxData.begin(),auxIt);
									*newInstanceData = mbInstanceData[instanceID];
									//assert(instanceData.begin()[instanceID].worldTform==newInstanceData->tform); TODO: later
									newInstanceData->padding0 = firstIndex;
									newInstanceData->padding1 = reinterpret_cast<const uint32_t&>(cdotIt->attribInfo[posAttrID]);
									newInstanceData->determinantSignBit = core::bitfieldInsert(
										newInstanceData->determinantSignBit,
										reinterpret_cast<const uint32_t&>(cdotIt->attribInfo[combinedNormalUVAttributeIx]),
										0u,31u
									);
									if (frontFaceIsCCW) // compensate for Nabla's default camera being left handed
										newInstanceData->determinantSignBit ^= 0x80000000u;

									auto& c = cullData.emplace_back();
									c.aabbMinEdge.x = batchAABB.MinEdge.X;
									c.aabbMinEdge.y = batchAABB.MinEdge.Y;
									c.aabbMinEdge.z = batchAABB.MinEdge.Z;
									c.batchInstanceGUID = batchInstanceGUID;
									c.aabbMaxEdge.x = batchAABB.MaxEdge.X;
									c.aabbMaxEdge.y = batchAABB.MaxEdge.Y;
									c.aabbMaxEdge.z = batchAABB.MaxEdge.Z;
									c.drawCommandGUID = drawCommandGUID;

									rrInstances.emplace_back() = rr->CreateInstance(rrShapes.back());
									rrInstances.back()->SetId(batchInstanceGUID);
									ext::RadeonRays::Manager::shapeSetTransform(rrInstances.back(),newInstanceData->tform);

									// set up scene bounds and lights
									if (i==0u)
									{
										if (mb==*meshBuffers.begin())
											m_sceneBound.addInternalBox(core::transformBoxEx(aabbMesh,newInstanceData->tform));
										const auto& emitter = auxIt->frontEmitter;
										if (emitter.type!=ext::MitsubaLoader::CElementEmitter::Type::INVALID)
										{
											assert(emitter.type==ext::MitsubaLoader::CElementEmitter::Type::AREA);

											SLight newLight(aabbMesh,newInstanceData->tform); // TODO: should be an OBB

											const float luma = newLight.computeLuma(emitter.area.radiance);
											const float weight = newLight.computeFluxBound(luma)*emitter.area.samplingWeight;
											if (weight<=FLT_MIN)
												continue;
											if (m_maxAreaLightLuma < luma)
												m_maxAreaLightLuma = luma;

											retval.lights.emplace_back(std::move(newLight));
											retval.lightPDF.push_back(weight);
										}
									}

									newInstanceData++;
								}
								for (auto j=thisShapeInstancesBeginIx; j!=rrInstances.size(); j++)
									rr->AttachShape(rrInstances[j]);
								cdotIt++;
								aabbsIt++;
							}
							//
							if (!mdiCall || pmbdIt->mdiParameterOffset!=mdiCall->mdiOffset+mdiCall->mdiCount)
							{
								mdiCall = &m_mdiDrawCalls.emplace_back();
								mdiCall->mdiOffset = pmbdIt->mdiParameterOffset;
								mdiCall->mdiCount = 0u;
							}
							mdiCall->mdiCount += pmbdIt->mdiParameterCount;
							//
							pmbdIt++;
						}
					}
				}
				printf("[INFO] Scene Bound: %f,%f,%f -> %f,%f,%f\n",
					m_sceneBound.MinEdge.X,
					m_sceneBound.MinEdge.Y,
					m_sceneBound.MinEdge.Z,
					m_sceneBound.MaxEdge.X,
					m_sceneBound.MaxEdge.Y,
					m_sceneBound.MaxEdge.Z
				);
				instanceDataDescPtr->buffer = {0u,cullData.size()*sizeof(ext::MitsubaLoader::instance_data_t)};
				instanceDataDescPtr->desc = std::move(newInstanceDataBuffer); // TODO: trim the buffer
				{
					auto gpump = core::make_smart_refctd_ptr<GPUMeshPacker>(m_driver,cpump.get());
					const auto& dataStore = gpump->getPackerDataStore();
					m_indexBuffer = dataStore.indexBuffer;
					// set up descriptor set for the inputs
					{
						for (auto i=0u; i<writeBound; i++)
						{
							recordInfoBuffer(infos[i],core::smart_refctd_ptr(dataStore.vertexBuffer));
							recordSSBOWrite(writes[i],infos+i,i);
						}
						recordInfoBuffer(infos[1],core::smart_refctd_ptr(m_indexBuffer));

						setDstSetOnAllWrites(m_additionalGlobalDS.get());
						m_driver->updateDescriptorSets(writeBound,writes,0u,nullptr);
					}
					// set up double buffering of MDI command buffers
					{
						m_indirectDrawBuffers[0] = dataStore.MDIDataBuffer;
						const auto mdiBufferSize = m_indirectDrawBuffers[0]->getSize();
						m_indirectDrawBuffers[1] = m_driver->createDeviceLocalGPUBufferOnDedMem(mdiBufferSize);
						m_driver->copyBuffer(m_indirectDrawBuffers[0].get(),m_indirectDrawBuffers[1].get(),0u,0u,mdiBufferSize);
					}
				}
			}
			m_cullPushConstants.maxDrawCommandCount = pmbd.back().mdiParameterOffset+pmbd.back().mdiParameterCount;
			m_cullPushConstants.maxGlobalInstanceCount = cullData.size();
		}

		// build TLAS with up to date transformations of instances
		rr->SetOption("bvh.sah.use_splits",1.f);
		rr->SetOption("bvh.builder","sah");
		// deinstance everything for great perf
		rr->SetOption("bvh.forceflat",1.f);
		rr->SetOption("acc.type","fatbvh");
		rr->Commit();
	}

	m_cullPushConstants.currentCommandBufferIx = 0x0u;
	m_cullWorkGroups = (m_cullPushConstants.maxGlobalInstanceCount-1u)/WORKGROUP_SIZE+1u;

	m_cullDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_cullDSLayout));
	m_perCameraRasterDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_perCameraRasterDSLayout));
	{
		recordInfoBuffer(infos[3],core::smart_refctd_ptr(m_indirectDrawBuffers[1]));
		recordInfoBuffer(infos[2],core::smart_refctd_ptr(m_indirectDrawBuffers[0]));
		recordInfoBuffer(infos[1],m_driver->createFilledDeviceLocalGPUBufferOnDedMem(m_cullPushConstants.maxGlobalInstanceCount*sizeof(CullData_t),cullData.data()));
		cullData.clear();
		recordInfoBuffer(infos[0],m_driver->createDeviceLocalGPUBufferOnDedMem(m_cullPushConstants.maxGlobalInstanceCount*sizeof(DrawData_t)));
		
		recordSSBOWrite(writes[0],infos+0,0u);
		recordSSBOWrite(writes[1],infos+1,1u);
		recordSSBOWrite(writes[2],infos+2,2u,2u);

		setDstSetOnAllWrites(m_perCameraRasterDS.get());
		m_driver->updateDescriptorSets(1u,writes,0u,nullptr);
		setDstSetOnAllWrites(m_cullDS.get());
		m_driver->updateDescriptorSets(3u,writes,0u,nullptr);
	}
	
	// TODO: after port to new API, use a converter which does not generate mip maps
	m_globalBackendDataDS = m_driver->getGPUObjectsFromAssets(&_globalBackendDataDS,&_globalBackendDataDS+1)->front();
	// make a shortened version of the globalBackendDataDS
	m_rasterInstanceDataDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_rasterInstanceDataDSLayout));
	{
		IGPUDescriptorSet::SCopyDescriptorSet copy = {};
		copy.dstSet = m_rasterInstanceDataDS.get();
		copy.srcSet = m_globalBackendDataDS.get();
		copy.srcBinding = 5u;
		copy.srcArrayElement = 0u;
		copy.dstBinding = 0u;
		copy.dstArrayElement = 0u;
		copy.count = 1u;
		m_driver->updateDescriptorSets(0u,nullptr,1u,&copy);
	}
	return retval;
}

void Renderer::initSceneNonAreaLights(Renderer::InitializationData& initData)
{
	core::smart_refctd_ptr<IGPUBuffer> paramsBuffer;
	{
		core::vector<core::vectorSIMDf> params({core::vectorSIMDf(0.0f,0.0f,0.0f,1.f)});
		for (const auto& emitter : m_globalMeta->m_global.m_emitters)
		{
			float weight = 0.f;
			switch (emitter.type)
			{
				case ext::MitsubaLoader::CElementEmitter::Type::CONSTANT:
				{
					params.front() += emitter.constant.radiance;
					break;
				}
				case ext::MitsubaLoader::CElementEmitter::Type::ENVMAP:
				{
					std::cout << "ENVMAP FOUND = " << std::endl;
					std::cout << "\tScale = " << emitter.envmap.scale << std::endl;
					std::cout << "\tGamma = " << emitter.envmap.gamma << std::endl;
					std::cout << "\tSamplingWeight = " << emitter.envmap.samplingWeight << std::endl;
					// LOAD file relative to the XML
					std::cout << "\tFileName = " << emitter.envmap.filename.svalue << std::endl;
					core::matrix3x4SIMD invTform;
					emitter.transform.matrix.extractSub3x4().getInverse(invTform);
					params.push_back(invTform.rows[0]);
					params.push_back(invTform.rows[1]);
					params.push_back(invTform.rows[2]);
					break;
				}
				case ext::MitsubaLoader::CElementEmitter::Type::INVALID:
					break;
				default:
				#ifdef _DEBUG
					assert(false);
				#endif
					// let's implement a new emitter type!
					//weight = emitter.unionType.samplingWeight;
					break;
			}
			if (weight==0.f)
				continue;
			
			//weight *= light.computeFlux(NAN);
			if (weight <= FLT_MIN)
				continue;

			//initData.lightPDF.push_back(weight);
			//initData.lights.push_back(light);
		}
		reinterpret_cast<uint32_t&>(params.front().w) = params.size()/3;
		paramsBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(core::vectorSIMDf)*params.size(),params.data());
	}
	const auto& envMapCPUImages = m_globalMeta->m_global.m_envMapImages;
	
	// don't touch this, 4x2 envmap is absolute minimum to have everything working
	uint32_t newWidth = 4;
	// create image
	{
		const auto colorFormat = asset::EF_E5B9G9R9_UFLOAT_PACK32;

		for (const auto& envmapCpuImage : envMapCPUImages)
		{
			const auto& extent = envmapCpuImage->getCreationParameters().extent;
			// I could upscale 2x2 if detecting a rotation, but should have used Cubemaps or Octahedral mapping instead 
			newWidth = core::max<uint32_t>(core::max<uint32_t>(extent.width,extent.height<<1u),newWidth);
		}
		constexpr uint32_t kMaxEnvMapSize = 16u << 10u;
		newWidth = core::roundUpToPoT<uint32_t>(core::min<uint32_t>(newWidth,kMaxEnvMapSize));

		// full mipchain would be `MSB+1` but we want it to stop at 4x2
		const auto mipLevels = core::findMSB(newWidth)-1;

		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = newWidth;
		imgInfo.extent.height = newWidth>>1u;
		imgInfo.extent.depth = 1u;

		imgInfo.mipLevels = mipLevels;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

		auto image = m_driver->createGPUImageOnDedMem(std::move(imgInfo),m_driver->getDeviceLocalGPUMemoryReqs());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = mipLevels;

		m_finalEnvmap = m_driver->createGPUImageView(std::move(imgViewInfo));
	}

	{
		// we don't have DLT on this branch
		auto sampler = m_driver->createGPUSampler(IGPUSampler::SParams{
			IGPUSampler::ETC_REPEAT,
			IGPUSampler::ETC_REPEAT,
			IGPUSampler::ETC_REPEAT,
			IGPUSampler::ETBC_FLOAT_OPAQUE_BLACK,
			IGPUSampler::ETF_LINEAR,
			IGPUSampler::ETF_LINEAR,
			IGPUSampler::ESMM_LINEAR,
			5,
			false,
			IGPUSampler::ECO_ALWAYS
		});
		core::vector<core::smart_refctd_ptr<IGPUImageView>> retainedViews;

		constexpr auto BindingCount = 3;
		// Initialize Pipeline and Resources for EnvMap Blending
		core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
		//
		{
			core::smart_refctd_ptr<IGPUSampler> samplers[MAX_SAMPLERS_COMPUTE];
			std::fill_n(samplers,MAX_SAMPLERS_COMPUTE,sampler);
			const IGPUDescriptorSetLayout::SBinding bindings[BindingCount] = {
				{0,EDT_STORAGE_BUFFER,1,ISpecializedShader::ESS_COMPUTE,nullptr},
				{1,EDT_STORAGE_IMAGE,1,ISpecializedShader::ESS_COMPUTE,nullptr},
				{2,EDT_COMBINED_IMAGE_SAMPLER,MAX_SAMPLERS_COMPUTE,ISpecializedShader::ESS_COMPUTE,samplers}
			};
			auto descriptorSetLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+BindingCount);
			
			pipeline = m_driver->createGPUComputePipeline(nullptr,
				m_driver->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(descriptorSetLayout)),
				gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../addEnvironmentEmitters.comp")
			);

			descriptorSet = m_driver->createGPUDescriptorSet(std::move(descriptorSetLayout));
		}
		//
		{
			std::unique_ptr<IGPUDescriptorSet::SDescriptorInfo[]> infos(new IGPUDescriptorSet::SDescriptorInfo[envMapCPUImages.size()+BindingCount-1u]);
				
			IGPUDescriptorSet::SWriteDescriptorSet writes[BindingCount];
			for (auto i=0; i<BindingCount; i++)
			{
				writes[i].dstSet = descriptorSet.get();
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].info = infos.get()+i;
			}
			writes[0].descriptorType = EDT_STORAGE_BUFFER;
			writes[1].descriptorType = EDT_STORAGE_IMAGE;
			writes[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
			writes[2].count = envMapCPUImages.size();

			infos[0].desc = paramsBuffer;
			infos[0].buffer = {0,paramsBuffer->getSize()};
			infos[1].desc = m_driver->createGPUImageView({
				static_cast<IGPUImageView::E_CREATE_FLAGS>(0u),
				m_finalEnvmap->getCreationParameters().image,
				IGPUImageView::ET_2D,
				asset::EF_R32_UINT, // need to view the RGB9E5 as R32UI to write to it
				{},{IGPUImage::EAF_COLOR_BIT,0u,1u,0u,1u}
			});
			infos[1].image = {nullptr,EIL_GENERAL};
			auto pInfoEnvmap = writes[2].info;
			auto envMapGPUImages = m_driver->getGPUObjectsFromAssets(envMapCPUImages.data(),envMapCPUImages.data()+envMapCPUImages.size());
			for(auto& image : *envMapGPUImages)
			{
				pInfoEnvmap->desc = retainedViews.emplace_back(m_driver->createGPUImageView({
					static_cast<IGPUImageView::E_CREATE_FLAGS>(0u),
					core::smart_refctd_ptr(image),
					IGPUImageView::ET_2D,
					image->getCreationParameters().format,
					{},{IGPUImage::EAF_COLOR_BIT,0u,1u,0u,1u}
				}));
				pInfoEnvmap->image = {nullptr,asset::EIL_SHADER_READ_ONLY_OPTIMAL};
				pInfoEnvmap++;
			}

			auto bindingCount = BindingCount;
			if (envMapCPUImages.empty())
				bindingCount--;
			m_driver->updateDescriptorSets(bindingCount,writes,0u,nullptr);
		}

		m_driver->bindComputePipeline(pipeline.get());
		m_driver->bindDescriptorSets(EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&descriptorSet.get(),nullptr);
		const auto xGroups = (newWidth-1u)/WORKGROUP_DIM+1u;
		m_driver->dispatch(xGroups,core::max(xGroups>>1u,1u),1u);

		// always needs doing after rendering
		// TODO: better filter and GPU accelerated
		COpenGLExtensionHandler::extGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}

	m_finalEnvmap->regenerateMipMapLevels();
	m_envMapImportanceSampling.initResources(m_finalEnvmap);
}

void Renderer::finalizeScene(Renderer::InitializationData& initData)
{
	if (initData.lights.empty())
		return;
// TODO: later
//	m_staticViewData.lightCount = initData.lights.size();

	const double weightSum = std::accumulate(initData.lightPDF.begin(),initData.lightPDF.end(),0.0);
	assert(weightSum>FLT_MIN);

	constexpr double UINT_MAX_DOUBLE = double(0x1ull<<32ull);
	const double weightSumRcp = UINT_MAX_DOUBLE/weightSum;

	auto outCDF = initData.lightCDF.begin();

	auto inPDF = initData.lightPDF.begin();
	double partialSum = *inPDF;

	auto computeCDF = [UINT_MAX_DOUBLE,weightSumRcp,&partialSum,&outCDF](uint32_t prevCDF) -> void
	{
		const double exactCDF = weightSumRcp*partialSum+double(FLT_MIN);
		if (exactCDF<UINT_MAX_DOUBLE)
			*outCDF = static_cast<uint32_t>(exactCDF);
		else
		{
			assert(exactCDF<UINT_MAX_DOUBLE+1.0);
			*outCDF = 0xdeadbeefu;
		}
	};

	computeCDF(0u);
	for (auto prevCDF=outCDF++; outCDF!=initData.lightCDF.end(); prevCDF=outCDF++)
	{
		partialSum += double(*(++inPDF));

		computeCDF(*prevCDF);
	}
}

core::smart_refctd_ptr<IGPUImageView> Renderer::createTexture(uint32_t width, uint32_t height, E_FORMAT format, uint32_t mipLevels, uint32_t layers)
{
	const auto real_layers = layers ? layers:1u;

	IGPUImage::SCreationParams imgparams;
	imgparams.extent = {width, height, 1u};
	imgparams.arrayLayers = real_layers;
	imgparams.flags = static_cast<IImage::E_CREATE_FLAGS>(0);
	imgparams.format = format;
	imgparams.mipLevels = mipLevels;
	imgparams.samples = IImage::ESCF_1_BIT;
	imgparams.type = IImage::ET_2D;

	IGPUImageView::SCreationParams viewparams;
	viewparams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0);
	viewparams.format = format;
	viewparams.image = m_driver->createDeviceLocalGPUImageOnDedMem(std::move(imgparams));
	viewparams.viewType = layers ? IGPUImageView::ET_2D_ARRAY:IGPUImageView::ET_2D;
	viewparams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
	viewparams.subresourceRange.baseArrayLayer = 0u;
	viewparams.subresourceRange.layerCount = real_layers;
	viewparams.subresourceRange.baseMipLevel = 0u;
	viewparams.subresourceRange.levelCount = mipLevels;

	return m_driver->createGPUImageView(std::move(viewparams));
}

core::smart_refctd_ptr<IGPUImageView> Renderer::createScreenSizedTexture(E_FORMAT format, uint32_t layers)
{
	return createTexture(m_staticViewData.imageDimensions[0], m_staticViewData.imageDimensions[1], format, 1u, layers);
}

core::smart_refctd_ptr<asset::ICPUBuffer> Renderer::SampleSequence::createCPUBuffer(uint32_t quantizedDimensions, uint32_t sampleCount)
{
	const size_t bytesize = SampleSequence::QuantizedDimensionsBytesize*quantizedDimensions*sampleCount;
	if (bytesize)
		return core::make_smart_refctd_ptr<asset::ICPUBuffer>(bytesize);
	else
		return nullptr;
}
void Renderer::SampleSequence::createBufferView(IVideoDriver* driver, core::smart_refctd_ptr<asset::ICPUBuffer>&& buff)
{
	auto gpubuf = driver->createFilledDeviceLocalGPUBufferOnDedMem(buff->getSize(),buff->getPointer());
	bufferView = driver->createGPUBufferView(gpubuf.get(),asset::EF_R32G32_UINT);
}
core::smart_refctd_ptr<ICPUBuffer> Renderer::SampleSequence::createBufferView(IVideoDriver* driver, uint32_t quantizedDimensions, uint32_t sampleCount)
{
	constexpr auto DimensionsPerQuanta = 3u;
	const auto dimensions = quantizedDimensions*DimensionsPerQuanta;
	core::OwenSampler sampler(dimensions,0xdeadbeefu);

	// Memory Order: 3 Dimensions, then multiple of sampling stragies per vertex, then depth, then sample ID
	auto buff = createCPUBuffer(quantizedDimensions,sampleCount);
	uint32_t(&pout)[][2] = *reinterpret_cast<uint32_t(*)[][2]>(buff->getPointer());
	// the horrible locality of iteration over output memory is caused by the fact that certain samplers like the 
	// Owen Scramble sampler, have a large cache which needs to be generated separately for each dimension.
	for (auto metadim=0u; metadim<quantizedDimensions; metadim++)
	{
		const auto trudim = metadim*DimensionsPerQuanta;
		for (uint32_t i=0; i<sampleCount; i++)
			pout[i*quantizedDimensions+metadim][0] = sampler.sample(trudim+0u,i);
		for (uint32_t i=0; i<sampleCount; i++)
			pout[i*quantizedDimensions+metadim][1] = sampler.sample(trudim+1u,i);
		for (uint32_t i=0; i<sampleCount; i++)
		{
			const auto sample = sampler.sample(trudim+2u,i);
			const auto out = pout[i*quantizedDimensions+metadim];
			out[0] &= 0xFFFFF800u;
			out[0] |= sample>>21;
			out[1] &= 0xFFFFF800u;
			out[1] |= (sample>>10)&0x07FFu;
		}
	}
	// upload sequence to GPU
	createBufferView(driver,core::smart_refctd_ptr(buff));
	// return for caching
	return buff;
}

// TODO: be able to fail
void Renderer::initSceneResources(SAssetBundle& meshes, nbl::io::path&& _sampleSequenceCachePath)
{
	deinitSceneResources();


	// set up Descriptor Sets
	{
		// captures m_globalBackendDataDS, creates m_indirectDrawBuffers, sets up m_mdiDrawCalls ranges
		// creates m_additionalGlobalDS and m_cullDS, sets m_cullPushConstants and m_cullWorkgroups, creates m_perCameraRasterDS
		auto initData = initSceneObjects(meshes);
		{
			initSceneNonAreaLights(initData);
			finalizeScene(initData);
		}

		//
		{
			// i know what I'm doing
			auto globalBackendDataDSLayout = core::smart_refctd_ptr<IGPUDescriptorSetLayout>(const_cast<IGPUDescriptorSetLayout*>(m_globalBackendDataDS->getLayout()));

			// cull
			{
				SPushConstantRange range{ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t)};
				m_cullPipelineLayout = m_driver->createGPUPipelineLayout(&range,&range+1u,core::smart_refctd_ptr(globalBackendDataDSLayout),core::smart_refctd_ptr(m_cullDSLayout),nullptr,nullptr);
			}

			SPushConstantRange raytracingCommonPCRange{ISpecializedShader::ESS_COMPUTE,0u,sizeof(RaytraceShaderCommonData_t)};
			// raygen
			{
				m_raygenPipelineLayout = m_driver->createGPUPipelineLayout(
					&raytracingCommonPCRange,&raytracingCommonPCRange+1u,
					core::smart_refctd_ptr(globalBackendDataDSLayout),
					core::smart_refctd_ptr(m_additionalGlobalDSLayout),
					core::smart_refctd_ptr(m_commonRaytracingDSLayout),
					core::smart_refctd_ptr(m_raygenDSLayout)
				);

				m_raygenDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_raygenDSLayout));
			}

			// closest hit
			{
				m_closestHitPipelineLayout = m_driver->createGPUPipelineLayout(
					&raytracingCommonPCRange,&raytracingCommonPCRange+1u,
					core::smart_refctd_ptr(globalBackendDataDSLayout),
					core::smart_refctd_ptr(m_additionalGlobalDSLayout),
					core::smart_refctd_ptr(m_commonRaytracingDSLayout),
					core::smart_refctd_ptr(m_closestHitDSLayout)
				);
			}

			// resolve
			{
				SPushConstantRange range{ISpecializedShader::ESS_COMPUTE,0u,sizeof(core::matrix3x4SIMD)+sizeof(nbl_glsl_RWMC_ReweightingParameters)};
				m_resolvePipelineLayout = m_driver->createGPUPipelineLayout(&range,&range+1,core::smart_refctd_ptr(m_resolveDSLayout));
				m_resolveDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_resolveDSLayout));
			}
			
			//
			auto setBufferInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, const core::smart_refctd_ptr<IGPUBuffer>& buffer) -> void
			{
				info->buffer.size = buffer->getSize();
				info->buffer.offset = 0u;
				info->desc = core::smart_refctd_ptr(buffer);
			};
			auto createFilledBufferAndSetUpInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, size_t size, const void* data)
			{
				auto buf = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(size,data);
				setBufferInfo(info,core::smart_refctd_ptr(buf));
				return buf;
			};
			auto createFilledBufferAndSetUpInfoFromVector = [createFilledBufferAndSetUpInfo](IGPUDescriptorSet::SDescriptorInfo* info, const auto& vector)
			{
				return createFilledBufferAndSetUpInfo(info,vector.size()*sizeof(decltype(*vector.data())),vector.data());
			};
			auto setDstSetAndDescTypesOnWrites = [](IGPUDescriptorSet* dstSet, IGPUDescriptorSet::SWriteDescriptorSet* writes, IGPUDescriptorSet::SDescriptorInfo* _infos, const std::initializer_list<asset::E_DESCRIPTOR_TYPE>& list, uint32_t baseBinding=0u)
			{
				auto typeIt = list.begin();
				for (auto i=0u; i<list.size(); i++)
				{
					writes[i].dstSet = dstSet;
					writes[i].binding = baseBinding+i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = *(typeIt++);
					writes[i].info = _infos+i;
				}
			};
			
			constexpr uint32_t MaxDescritorUpdates = 2u;
			IGPUDescriptorSet::SDescriptorInfo infos[MaxDescritorUpdates];
			IGPUDescriptorSet::SWriteDescriptorSet writes[MaxDescritorUpdates];

			size_t lightCDF_BufferSize = 0u;
			size_t lights_BufferSize = 0u;

			// set up rest of m_additionalGlobalDS
			if(initData.lights.empty())
			{
				std::cout << "\n[ERROR] No supported lights found in the scene.";
			}
			else
			{
				auto lightCDFBuffer = createFilledBufferAndSetUpInfoFromVector(infos+0,initData.lightCDF);
				auto lightsBuffer = createFilledBufferAndSetUpInfoFromVector(infos+1,initData.lights);
				lightCDF_BufferSize = lightCDFBuffer->getSize();
				lights_BufferSize = lightsBuffer->getSize();
				setDstSetAndDescTypesOnWrites(m_additionalGlobalDS.get(),writes,infos,{EDT_STORAGE_BUFFER,EDT_STORAGE_BUFFER},3u);
				m_driver->updateDescriptorSets(2u,writes,0u,nullptr);
			}

			std::cout << "\nScene Resources Initialized:" << std::endl;
			std::cout << "\tlightCDF = " << lightCDF_BufferSize << " bytes" << std::endl;
			std::cout << "\tlights = " << lights_BufferSize << " bytes" << std::endl;
			std::cout << "\tindexBuffer = " << m_indexBuffer->getSize() << " bytes" << std::endl;
			for (auto i=0u; i<2u; i++)
				std::cout << "\tIndirect Draw Buffers[" << i << "] = " << m_indirectDrawBuffers[i]->getSize() << " bytes" << std::endl;
		}
		
		// load sample cache
		{
			core::smart_refctd_ptr<ICPUBuffer> cachebuff;
			uint32_t cachedQuantizedDimensions=0u,cachedSampleCount=0u;
			{
				sampleSequenceCachePath = std::move(_sampleSequenceCachePath);
				io::IReadFile* cacheFile = m_assetManager->getFileSystem()->createAndOpenFile(sampleSequenceCachePath);
				if (cacheFile)
				{
					cacheFile->read(&cachedQuantizedDimensions,sizeof(cachedQuantizedDimensions));
					if (cachedQuantizedDimensions)
					{
						cachedSampleCount = (cacheFile->getSize()-cacheFile->getPos())/(cachedQuantizedDimensions*SampleSequence::QuantizedDimensionsBytesize);
						cachebuff = sampleSequence.createCPUBuffer(cachedQuantizedDimensions,cachedSampleCount);
						if (cachebuff)
							cacheFile->read(cachebuff->getPointer(),cachebuff->getSize());
					}
					cacheFile->drop();
				}
			}
			// lets keep path length within bounds of sanity
			constexpr auto MaxPathDepth = 255u;
			if (maxPathDepth==0)
			{
				printf("[ERROR] No suppoerted Integrator found in the Mitsuba XML, setting default.\n");
				maxPathDepth = DefaultPathDepth;
			}
			else if (maxPathDepth>MaxPathDepth)
			{
				printf("[WARNING] Path Depth %d greater than maximum supported, clamping to %d\n",maxPathDepth,MaxPathDepth);
				maxPathDepth = MaxPathDepth;
			}
			const uint32_t quantizedDimensions = SampleSequence::computeQuantizedDimensions(maxPathDepth);
			// The primary limiting factor is the precision of turning a fixed point grid sample to IEEE754 32bit float in the [0,1] range.
			// Mantissa is only 23 bits, and primary sample space low discrepancy sequence will start to produce duplicates
			// near 1.0 with exponent -1 after the sample count passes 2^24 elements.
			// Another limiting factor is our encoding of sample sequences, we only use 21bits per channel, so no duplicates till 2^21 samples.
			maxSensorSamples = core::min(0x1u<<21u,maxSensorSamples);
			if (cachedQuantizedDimensions>=quantizedDimensions && cachedSampleCount>=maxSensorSamples)
				sampleSequence.createBufferView(m_driver,std::move(cachebuff));
			else
			{
				printf("[INFO] Generating Low Discrepancy Sample Sequence Cache, please wait...\n");
				cachebuff = sampleSequence.createBufferView(m_driver,quantizedDimensions,maxSensorSamples);
				// save sequence
				io::IWriteFile* cacheFile = m_assetManager->getFileSystem()->createAndWriteFile(sampleSequenceCachePath);
				if (cacheFile)
				{
					cacheFile->write(&quantizedDimensions,sizeof(quantizedDimensions));
					cacheFile->write(cachebuff->getPointer(),cachebuff->getSize());
					cacheFile->drop();
				}
			}
			std::cout << "\tmaxPathDepth = " << maxPathDepth << std::endl;
			std::cout << "\tnoRussianRouletteDepth = " << noRussianRouletteDepth << std::endl;
			std::cout << "\thideEnvironment = " << hideEnvironment << std::endl;
			std::cout << "\tmaxSamples = " << maxSensorSamples << std::endl;
		}
	}
	std::cout << std::endl;
}

void Renderer::deinitSceneResources()
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);

	glFinish();

	m_resolveDS = nullptr;
	m_raygenDS = nullptr;
	m_additionalGlobalDS = nullptr;
	m_rasterInstanceDataDS = nullptr;
	m_globalBackendDataDS = nullptr;

	m_perCameraRasterDS = nullptr;
	
	m_cullPipelineLayout = nullptr;
	m_raygenPipelineLayout = nullptr;
	m_closestHitPipelineLayout = nullptr;
	m_resolvePipelineLayout = nullptr;

	m_cullWorkGroups = 0u;
	m_cullPushConstants = {core::matrix4SIMD(),1.f,0u,0u,0u};
	m_cullDS = nullptr;
	m_mdiDrawCalls.clear();
	m_indirectDrawBuffers[1] = m_indirectDrawBuffers[0] = nullptr;
	m_indexBuffer = nullptr;

	m_raytraceCommonData = { 0.f,0u,0u,0u,core::matrix3x4SIMD() };
	m_sceneBound = core::aabbox3df(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
	m_maxAreaLightLuma = 0.f;
	
	m_finalEnvmap = nullptr;
	m_envMapImportanceSampling.deinitResources();
	m_staticViewData = {{0u,0u},0u,0u,0u,0u,false,core::infinity<float>(),{}};

	auto rr = m_rrManager->getRadeonRaysAPI();
	rr->DetachAll();
	for (auto instance : rrInstances)
	{
		rr->DeleteShape(instance);
	}
	rrInstances.clear();

	for (auto shape : rrShapes)
		rr->DeleteShape(shape);
	rrShapes.clear();

	maxPathDepth = DefaultPathDepth;
	noRussianRouletteDepth = 5u;
	hideEnvironment = false;
	maxSensorSamples = MaxFreeviewSamples;
}

void Renderer::deinitRenderer()
{
	m_driver = nullptr;
	m_smgr = nullptr;
	m_assetManager = nullptr;
	m_rrManager = nullptr;
}

void Renderer::finalizeDeferredDenoise()
{
	if (!m_deferDenoise)
		return;

	m_deferDenoiseFile.close();
	std::cout << "\n---[DENOISER_BEGIN]---" << std::endl;
	std::system(DEFER_DENOISE_HOOK_FILE_NAME);
	std::cout << "\n---[DENOISER_END]---" << std::endl;
	std::remove(DEFER_DENOISE_HOOK_FILE_NAME);
}

void Renderer::initScreenSizedResources(
	const uint32_t width, const uint32_t height,
	const float envMapRegularizationFactor,
	int32_t cascadeCount,
	float cascadeLuminanceBase,
	float cascadeLuminanceStart,
	const float Emin,
	const nbl::core::vector<nbl::core::vectorSIMDf>& clipPlanes
)
{
	float maxEmitterRadianceLuma;
	bool enableRIS = m_envMapImportanceSampling.computeWarpMap(envMapRegularizationFactor,m_staticViewData.envMapPDFNormalizationFactor,maxEmitterRadianceLuma);
	if (maxEmitterRadianceLuma<m_maxAreaLightLuma)
		maxEmitterRadianceLuma = m_maxAreaLightLuma;
	if (maxEmitterRadianceLuma<Emin)
		maxEmitterRadianceLuma = Emin+0.1234567f;

	constexpr int32_t MinCascades = 2; // due to impl details
	constexpr int32_t MaxCascades = 32; // sane limit
	const float RGB19E7_MaxLuma = std::exp2(63.f);
	if (cascadeCount<MinCascades) // rwmc OFF, store everything to cascade 0
	{
		// original idea was to create 2 cascades where the first starts so low that every sample gets added to it. But now we just do 1
		cascadeCount = 0;
		cascadeLuminanceBase = std::exp2(16.f); // just some constant to space the cascades apart
		cascadeLuminanceStart = RGB19E7_MaxLuma;
		std::cout << "Re-Weighting Monte Carlo = DISABLED" << std::endl;
	}
	else
	{
		cascadeCount = core::min(cascadeCount,MaxCascades);
		const float cascadeSegmentCount = cascadeCount-1;
		// base is the power increment between each successive cascade, first cascade starts at Emin or 1/cascadeLuminanceBase^segmentCount scaled to max emitter radiance
		const bool baseIsKnown = cascadeLuminanceBase>std::numeric_limits<float>::min();
		if (core::isnan<float>(cascadeLuminanceStart))
			cascadeLuminanceStart = baseIsKnown ? (maxEmitterRadianceLuma*std::pow(cascadeLuminanceBase,-cascadeSegmentCount)):Emin;
		// rationale, we don't have NEE and BRDF importance sampling samples with throughput <= 1.0
		// However we have RIS, and that can complicate this assumption a bit
		if (!baseIsKnown)
			cascadeLuminanceBase = core::max(std::pow(maxEmitterRadianceLuma/cascadeLuminanceStart,1.f/cascadeSegmentCount),1.0625f);
		std::cout << "Re-Weighting Monte Carlo = ENABLED [cascadeCount: "<<cascadeCount<<", start: "<<cascadeLuminanceStart<<", base: "<<cascadeLuminanceBase<<"]" << std::endl;
	}

	m_staticViewData.cascadeParams = nbl_glsl_RWMC_computeCascadeParameters(cascadeCount,cascadeLuminanceStart,cascadeLuminanceBase);
	m_staticViewData.imageDimensions[0] = width;
	m_staticViewData.imageDimensions[1] = height;
	m_rcpPixelSize = { 2.f/float(m_staticViewData.imageDimensions[0]),-2.f/float(m_staticViewData.imageDimensions[1]) };

	// figure out dispatch sizes
	m_raygenWorkGroups[0] = (m_staticViewData.imageDimensions[0]-1u)/WORKGROUP_DIM+1u;
	m_raygenWorkGroups[1] = (m_staticViewData.imageDimensions[1]-1u)/WORKGROUP_DIM+1u;

	const auto renderPixelCount = m_staticViewData.imageDimensions[0]*m_staticViewData.imageDimensions[1];
	// figure out how much Samples Per Pixel Per Dispatch we can afford
	size_t scrambleBufferSize=0u;
	size_t raygenBufferSize=0u,intersectionBufferSize=0u;
	{
		m_staticViewData.maxPathDepth = maxPathDepth;
		m_staticViewData.noRussianRouletteDepth = noRussianRouletteDepth;
		m_staticViewData.hideEnvmap = hideEnvironment;

		uint32_t _maxRaysPerDispatch = 0u;
		auto setRayBufferSizes = [renderPixelCount,this,&_maxRaysPerDispatch,&raygenBufferSize,&intersectionBufferSize](uint32_t sampleMultiplier) -> void
		{
			m_staticViewData.samplesPerPixelPerDispatch = sampleMultiplier;

			const size_t minimumSampleCountPerDispatch = static_cast<size_t>(renderPixelCount)*getSamplesPerPixelPerDispatch();
			_maxRaysPerDispatch = static_cast<uint32_t>(minimumSampleCountPerDispatch);
			const auto doubleBufferSampleCountPerDispatch = minimumSampleCountPerDispatch*2ull;

			raygenBufferSize = doubleBufferSampleCountPerDispatch*sizeof(::RadeonRays::ray);
			intersectionBufferSize = doubleBufferSampleCountPerDispatch*sizeof(::RadeonRays::Intersection);
		};
		// see how much we can bump the sample count per raster pass
		{
			uint32_t sampleMultiplier = 0u;
			const auto maxSSBOSize = core::min(m_driver->getMaxSSBOSize(),256u<<20);
			while (sampleMultiplier<0x10000u && raygenBufferSize<=maxSSBOSize && intersectionBufferSize<=maxSSBOSize)
				setRayBufferSizes(++sampleMultiplier);
			if (sampleMultiplier==1u)
				setRayBufferSizes(sampleMultiplier);
			printf("[INFO] Using %d samples (per pixel) per dispatch\n",getSamplesPerPixelPerDispatch());
		}
	}
	m_staticViewData.sampleSequenceStride = SampleSequence::computeQuantizedDimensions(maxPathDepth);
	auto stream = std::ofstream("runtime_defines.glsl");

	for (auto i=0; i<ext::MitsubaLoader::CElementSensor::MaxClipPlanes; i++)
	{
		if (i<clipPlanes.size())
		{
			glEnable(GL_CLIP_DISTANCE0+i);
			stream << "#define CLIP_PLANE_" << i << " vec4(" << clipPlanes[i].x << "," << clipPlanes[i].y << "," << clipPlanes[i].z << "," << clipPlanes[i].w << ")\n";
		}
		else
			glDisable(GL_CLIP_DISTANCE0+i);
	}

	stream << "#define _NBL_EXT_MITSUBA_LOADER_VT_STORAGE_VIEW_COUNT " << m_globalMeta->m_global.getVTStorageViewCount() << "\n"
		<< m_globalMeta->m_global.m_materialCompilerGLSL_declarations
		<< "#ifndef MAX_RAYS_GENERATED\n"
		<< "#	define MAX_RAYS_GENERATED " << getSamplesPerPixelPerDispatch() << "\n"
		<< "#endif\n";

	if(!enableRIS)
		stream << "#define ONLY_BXDF_SAMPLING\n";

	stream.close();
	
	compileShadersFuture = std::async(std::launch::async, [&]()
	{
		// cull
		m_cullGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../cull.comp");

		// visbuffer
		m_vertGPUShader = gpuSpecializedShaderFromFile(m_assetManager, m_driver, "../fillVisBuffer.vert");

		// raygen
		m_raygenGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../raygen.comp");

		// closest hit
		m_closestHitGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../closestHit.comp");

		// resolve
		m_resolveGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,m_useDenoiser ? "../resolveForDenoiser.comp":"../resolve.comp");
		
		bool success = m_cullGPUShader && m_raygenGPUShader && m_closestHitGPUShader && m_resolveGPUShader;
		return success;
	});

	auto setBufferInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, const core::smart_refctd_ptr<IGPUBuffer>& buffer) -> void
	{
		info->buffer.size = buffer->getSize();
		info->buffer.offset = 0u;
		info->desc = core::smart_refctd_ptr(buffer);
	};

	auto createFilledBufferAndSetUpInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, size_t size, const void* data)
	{
		auto buf = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(size,data);
		setBufferInfo(info,core::smart_refctd_ptr(buf));
		return buf;
	};
	auto createFilledBufferAndSetUpInfoFromStruct = [createFilledBufferAndSetUpInfo](IGPUDescriptorSet::SDescriptorInfo* info, const auto& _struct)
	{
		return createFilledBufferAndSetUpInfo(info,sizeof(_struct),&_struct);
	};
	auto createFilledBufferAndSetUpInfoFromVector = [createFilledBufferAndSetUpInfo](IGPUDescriptorSet::SDescriptorInfo* info, const auto& vector)
	{
		return createFilledBufferAndSetUpInfo(info,vector.size()*sizeof(decltype(*vector.data())),vector.data());
	};
	auto setImageInfo = [](IGPUDescriptorSet::SDescriptorInfo* info, const asset::E_IMAGE_LAYOUT imageLayout, core::smart_refctd_ptr<IGPUImageView>&& imageView) -> void
	{
		info->image.imageLayout = imageLayout;
		info->image.sampler = nullptr; // storage image dont have samplers, and the combined sampler image views we have all use immutable samplers
		info->desc = std::move(imageView);
	};
	auto createEmptyInteropBufferAndSetUpInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, InteropBuffer& interopBuffer, size_t size) -> void
	{
		if (static_cast<COpenGLDriver*>(m_driver)->runningInRenderdoc()) // makes Renderdoc capture the modifications done by OpenCL
		{
			interopBuffer.buffer = m_driver->createUpStreamingGPUBufferOnDedMem(size);
//			interopBuffer.buffer->getBoundMemory()->mapMemoryRange(IDriverMemoryAllocation::EMCAF_READ_AND_WRITE,{0u,size});
		}
		else
			interopBuffer.buffer = m_driver->createDeviceLocalGPUBufferOnDedMem(size);
		interopBuffer.asRRBuffer = m_rrManager->linkBuffer(interopBuffer.buffer.get(), CL_MEM_READ_ONLY);

		info->buffer.size = size;
		info->buffer.offset = 0u;
		info->desc = core::smart_refctd_ptr(interopBuffer.buffer);
	};
	auto setDstSetAndDescTypesOnWrites = [](IGPUDescriptorSet* dstSet, IGPUDescriptorSet::SWriteDescriptorSet* writes, IGPUDescriptorSet::SDescriptorInfo* _infos, const std::initializer_list<asset::E_DESCRIPTOR_TYPE>& list, uint32_t baseBinding=0u)
	{
		auto typeIt = list.begin();
		for (auto i=0u; i<list.size(); i++)
		{
			writes[i].dstSet = dstSet;
			writes[i].binding = baseBinding+i;
			writes[i].arrayElement = 0u;
			writes[i].count = 1u;
			writes[i].descriptorType = *(typeIt++);
			writes[i].info = _infos+i;
		}
	};

	// create out screen-sized textures
	m_accumulation = createScreenSizedTexture(EF_R32G32_UINT,(cascadeCount+1u)*m_staticViewData.samplesPerPixelPerDispatch); // one more (first) layer because of accumulation metadata for a path
	m_albedoAcc = createScreenSizedTexture(EF_R32_UINT,m_staticViewData.samplesPerPixelPerDispatch);
	m_normalAcc = createScreenSizedTexture(EF_R32_UINT,m_staticViewData.samplesPerPixelPerDispatch);
	m_maskAcc = createScreenSizedTexture(EF_R16_UNORM,m_staticViewData.samplesPerPixelPerDispatch);
	m_tonemapOutput = createScreenSizedTexture(EF_R16G16B16A16_SFLOAT);
	m_albedoRslv = createScreenSizedTexture(EF_A2B10G10R10_UNORM_PACK32);
	m_normalRslv = createScreenSizedTexture(EF_R16G16B16A16_SFLOAT);

	constexpr uint32_t MaxDescritorUpdates = 11u;
	IGPUDescriptorSet::SDescriptorInfo infos[MaxDescritorUpdates];
	IGPUDescriptorSet::SWriteDescriptorSet writes[MaxDescritorUpdates];
	
	auto warpMap = m_envMapImportanceSampling.getWarpMapImageView();
	auto lumaMap = m_envMapImportanceSampling.getLuminanceImageView();
	
	// set up m_commonRaytracingDS
	core::smart_refctd_ptr<IGPUBuffer> _staticViewDataBuffer;
	size_t staticViewDataBufferSize=0u;
	{
		_staticViewDataBuffer = createFilledBufferAndSetUpInfoFromStruct(infos+0,m_staticViewData);
		staticViewDataBufferSize = _staticViewDataBuffer->getSize();
		infos[1].desc = sampleSequence.getBufferView();
		setImageInfo(infos+2,asset::EIL_GENERAL,core::smart_refctd_ptr(m_accumulation));
		setImageInfo(infos+5,asset::EIL_GENERAL,core::smart_refctd_ptr(m_albedoAcc));
		setImageInfo(infos+6,asset::EIL_GENERAL,core::smart_refctd_ptr(m_normalAcc));
		setImageInfo(infos+7,asset::EIL_GENERAL,core::smart_refctd_ptr(m_maskAcc));

		// envmap
		{
			setImageInfo(infos+8,asset::EIL_GENERAL,core::smart_refctd_ptr(m_finalEnvmap));
			ISampler::SParams samplerParams = { ISampler::ETC_REPEAT, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			infos[8].image.sampler = m_driver->createGPUSampler(samplerParams);
			infos[8].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}
		// warpmap
		{
			setImageInfo(infos+9,asset::EIL_GENERAL,core::smart_refctd_ptr(warpMap));
			ISampler::SParams samplerParams = { ISampler::ETC_REPEAT, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			infos[9].image.sampler = m_driver->createGPUSampler(samplerParams);
			infos[9].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}
		
		IGPUDescriptorSet::SDescriptorInfo luminanceDescriptorInfo = {};
		// luminance mip maps
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_BORDER, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, 0u, false, ECO_ALWAYS };
			auto sampler = m_driver->createGPUSampler(samplerParams);

			luminanceDescriptorInfo.desc = lumaMap;
			luminanceDescriptorInfo.image.sampler = sampler;
			luminanceDescriptorInfo.image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
		}

		createEmptyInteropBufferAndSetUpInfo(infos+3,m_rayBuffer[0],raygenBufferSize);
		setBufferInfo(infos+4,m_rayCountBuffer);
			
		for (auto i=0u; i<2u; i++)
			m_commonRaytracingDS[i] = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_commonRaytracingDSLayout));

		constexpr auto descriptorUpdateCount = 11u;
		setDstSetAndDescTypesOnWrites(m_commonRaytracingDS[0].get(),writes,infos,{
			EDT_UNIFORM_BUFFER,
			EDT_UNIFORM_TEXEL_BUFFER,
			EDT_STORAGE_IMAGE,
			EDT_STORAGE_BUFFER,
			EDT_STORAGE_BUFFER,
			EDT_STORAGE_IMAGE,
			EDT_STORAGE_IMAGE,
			EDT_STORAGE_IMAGE,
			EDT_COMBINED_IMAGE_SAMPLER,
			EDT_COMBINED_IMAGE_SAMPLER,
		});
		
		// Set last write
		writes[10].binding = 10u;
		writes[10].arrayElement = 0u;
		writes[10].count = 1u;
		writes[10].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		writes[10].dstSet = m_commonRaytracingDS[0].get();
		writes[10].info = &luminanceDescriptorInfo;

		m_driver->updateDescriptorSets(descriptorUpdateCount,writes,0u,nullptr);
		// set up second DS
		createEmptyInteropBufferAndSetUpInfo(infos+3,m_rayBuffer[1],raygenBufferSize);
		for (auto i=0u; i<descriptorUpdateCount; i++)
			writes[i].dstSet = m_commonRaytracingDS[1].get();
		m_driver->updateDescriptorSets(descriptorUpdateCount,writes,0u,nullptr);
	}

	// set up m_raygenDS
		{
			constexpr auto ScrambleStateChannels = 2u;
			auto tmpBuff = m_driver->createCPUSideGPUVisibleGPUBufferOnDedMem(sizeof(uint32_t)*ScrambleStateChannels*renderPixelCount);
			// generate (maybe let's improve the scramble key beginning distribution)
			{
				core::RandomSampler rng(0xbadc0ffeu);
				auto it = reinterpret_cast<uint32_t*>(tmpBuff->getBoundMemory()->mapMemoryRange(
					IDriverMemoryAllocation::EMCAF_WRITE,
					IDriverMemoryAllocation::MemoryRange(0u,tmpBuff->getSize())
				));
				for (auto end=it+ScrambleStateChannels*renderPixelCount; it!=end; it++)
					*it = rng.nextSample();
				tmpBuff->getBoundMemory()->unmapMemory();
			}
			scrambleBufferSize = tmpBuff->getSize();
			// upload
			IGPUImage::SBufferCopy region;
			//region.imageSubresource.aspectMask = ;
			region.imageSubresource.baseArrayLayer = 0u;
			region.imageSubresource.layerCount = 1u;
			region.imageExtent = {m_staticViewData.imageDimensions[0],m_staticViewData.imageDimensions[1],1u};
			auto scrambleKeys = createScreenSizedTexture(EF_R32G32_UINT);
			m_driver->copyBufferToImage(tmpBuff.get(),scrambleKeys->getCreationParameters().image.get(),1u,&region);
			setImageInfo(infos+0,asset::EIL_SHADER_READ_ONLY_OPTIMAL,std::move(scrambleKeys));
		}
		setImageInfo(infos+2,asset::EIL_GENERAL,core::smart_refctd_ptr(m_tonemapOutput));

		setDstSetAndDescTypesOnWrites(m_raygenDS.get(),writes,infos,{
			EDT_COMBINED_IMAGE_SAMPLER,
			EDT_COMBINED_IMAGE_SAMPLER,
			EDT_STORAGE_IMAGE
		});
		m_driver->updateDescriptorSets(3u,writes,0u,nullptr);

	// set up m_closestHitDS
	for (auto i=0u; i<2u; i++)
	{
		const auto other = i^0x1u;
		infos[0u].desc = m_rayBuffer[other].buffer;
		infos[0u].buffer.offset = 0u;
		infos[0u].buffer.size = m_rayBuffer[other].buffer->getSize();
		createEmptyInteropBufferAndSetUpInfo(infos+1,m_intersectionBuffer[other],intersectionBufferSize);
				
		m_closestHitDS[i] = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_closestHitDSLayout));

		setDstSetAndDescTypesOnWrites(m_closestHitDS[i].get(),writes,infos,{EDT_STORAGE_BUFFER,EDT_STORAGE_BUFFER});
		m_driver->updateDescriptorSets(2u,writes,0u,nullptr);
	}

	// set up m_resolveDS
	{
		infos[0].buffer = {0u,_staticViewDataBuffer->getSize()};
		infos[0].desc = std::move(_staticViewDataBuffer);
		setImageInfo(infos+1,asset::EIL_GENERAL,core::smart_refctd_ptr(m_accumulation));
		core::smart_refctd_ptr<IGPUImageView> albedoSamplerView;
		{
			IGPUImageView::SCreationParams viewparams = m_albedoAcc->getCreationParameters();
			viewparams.format = EF_A2B10G10R10_UNORM_PACK32;
			albedoSamplerView = m_driver->createGPUImageView(std::move(viewparams));
		}
		setImageInfo(infos+2,asset::EIL_GENERAL,std::move(albedoSamplerView));
		setImageInfo(infos+3,asset::EIL_GENERAL,core::smart_refctd_ptr(m_normalAcc));
		setImageInfo(infos+4,asset::EIL_GENERAL,core::smart_refctd_ptr(m_maskAcc));
		setImageInfo(infos+5,asset::EIL_GENERAL,core::smart_refctd_ptr(m_tonemapOutput));
		core::smart_refctd_ptr<IGPUImageView> albedoStorageView;
		{
			IGPUImageView::SCreationParams viewparams = m_albedoRslv->getCreationParameters();
			viewparams.format = EF_R32_UINT;
			albedoStorageView = m_driver->createGPUImageView(std::move(viewparams));
		}
		setImageInfo(infos+6,asset::EIL_GENERAL,std::move(albedoStorageView));
		setImageInfo(infos+7,asset::EIL_GENERAL,core::smart_refctd_ptr(m_normalRslv));
				
		setDstSetAndDescTypesOnWrites(m_resolveDS.get(),writes,infos,{
			EDT_UNIFORM_BUFFER,
			EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER,
			EDT_STORAGE_IMAGE,EDT_STORAGE_IMAGE,EDT_STORAGE_IMAGE
		});
	}
	m_driver->updateDescriptorSets(8u,writes,0u,nullptr);


	m_colorBuffer = m_driver->addFrameBuffer();
	m_colorBuffer->attach(EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(m_tonemapOutput));

	std::cout << "\nScreen Sized Resources have been initialized (" << width << "x" << height << ")" << std::endl;
	std::cout << "\tStaticViewData = " << staticViewDataBufferSize << " bytes" << std::endl;
	std::cout << "\tScrambleBuffer = " << scrambleBufferSize << " bytes" << std::endl;
	std::cout << "\tSampleSequence = " << sampleSequence.getBufferView()->getByteSize() << " bytes" << std::endl;
	std::cout << "\tRayCount Buffer = " << m_rayCountBuffer->getSize() << " bytes" << std::endl;
	for (auto i=0u; i<2u; i++)
		std::cout << "\tIntersection Buffer[" << i << "] = " << m_intersectionBuffer[i].buffer->getSize() << " bytes" << std::endl;
	for (auto i=0u; i<2u; i++)
		std::cout << "\tRay Buffer[" << i << "] = " << m_rayBuffer[i].buffer->getSize() << " bytes" << std::endl;
	std::cout << std::endl;
}

void Renderer::takeAndSaveScreenShot(const std::filesystem::path& screenshotFilePath, bool denoise, const DenoiserArgs& denoiserArgs)
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);

	glFinish();

	// we always decode to 16bit HDR because thats what the denoiser takes
	// if we save to PNG instead of EXR, it will be converted and clamped once more automagically
	const asset::E_FORMAT format = asset::EF_R16G16B16A16_SFLOAT;

	auto filename_wo_ext = screenshotFilePath;
	filename_wo_ext.replace_extension();
	if (m_tonemapOutput)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_tonemapOutput.get(),filename_wo_ext.string()+".exr",format);
	if (m_albedoRslv)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_albedoRslv.get(),filename_wo_ext.string()+"_albedo.exr",format);
	if (m_normalRslv)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_normalRslv.get(),filename_wo_ext.string()+"_normal.exr",format);

	if (!denoise)
		return;

	const std::string defaultBloomFile = "../../media/kernels/physical_flare_512.exr";
	const std::string defaultTonemapperArgs = "ACES=0.4,0.8";
	constexpr auto defaultBloomScale = 0.1f;
	constexpr auto defaultBloomIntensity = 0.1f;
	auto bloomFilePathStr = (denoiserArgs.bloomFilePath.string().empty()) ? defaultBloomFile : denoiserArgs.bloomFilePath.string();
	auto bloomScale = (denoiserArgs.bloomScale == 0.0f) ? defaultBloomScale : denoiserArgs.bloomScale;
	auto bloomIntensity = (denoiserArgs.bloomIntensity == 0.0f) ? defaultBloomIntensity : denoiserArgs.bloomIntensity;
	auto tonemapperArgs = (denoiserArgs.tonemapperArgs.empty()) ? defaultTonemapperArgs : denoiserArgs.tonemapperArgs;

	std::ostringstream denoiserCmd;
	// 1.ColorFile 2.AlbedoFile 3.NormalFile 4.BloomPsfFilePath(STRING) 5.BloomScale(FLOAT) 6.BloomIntensity(FLOAT) 7.TonemapperArgs(STRING)
	denoiserCmd << "call ../denoiser_hook.bat";
	denoiserCmd << " \"" << filename_wo_ext.string() << ".exr" << "\"";
	denoiserCmd << " \"" << filename_wo_ext.string() << "_albedo.exr" << "\"";
	denoiserCmd << " \"" << filename_wo_ext.string() << "_normal.exr" << "\"";
	denoiserCmd << " \"" << bloomFilePathStr << "\"";
	denoiserCmd << " " << bloomScale;
	denoiserCmd << " " << bloomIntensity;
	denoiserCmd << " " << "\"" << tonemapperArgs << "\"";

	if (m_deferDenoise)
	{
		// TODO[Przemek]: what to do when m_deferDenoiseFile is not open? crash or log error?
		const bool isDeferDenoiseFileOpen = m_deferDenoiseFile.is_open();
		assert(isDeferDenoiseFileOpen);

		if (isDeferDenoiseFileOpen)
			m_deferDenoiseFile << denoiserCmd.str() << std::endl;
	}
	else
	{
		// NOTE/TODO/FIXME : Do as I say, not as I do
		// https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152177
		std::cout << "\n---[DENOISER_BEGIN]---" << std::endl;
		std::system(denoiserCmd.str().c_str());
		std::cout << "\n---[DENOISER_END]---" << std::endl;
	}

}

void Renderer::denoiseCubemapFaces(
	std::filesystem::path filePaths[6],
	const std::string& mergedFileName,
	int32_t cropOffsetX, int32_t cropOffsetY, int32_t cropWidth, int32_t cropHeight,
	const DenoiserArgs& denoiserArgs)
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);

	glFinish();

	std::string renderFilePaths[6] = {};
	std::string albedoFilePaths[6] = {};
	std::string normalFilePaths[6] = {};
	for(uint32_t i = 0; i < 6; ++i)
		renderFilePaths[i] = filePaths[i].replace_extension().string() + ".exr";
	for(uint32_t i = 0; i < 6; ++i)
		albedoFilePaths[i] = filePaths[i].replace_extension().string() + "_albedo.exr";
	for(uint32_t i = 0; i < 6; ++i)
		normalFilePaths[i] = filePaths[i].replace_extension().string() + "_normal.exr";
	
	std::filesystem::path mergedCubeMapOutputPath = filePaths[0].parent_path();

	std::filesystem::path mergedRenderFilePath = mergedCubeMapOutputPath / std::filesystem::path(mergedFileName + ".exr");
	std::filesystem::path mergedAlbedoFilePath = mergedCubeMapOutputPath / std::filesystem::path(mergedFileName + "_albedo.exr");
	std::filesystem::path mergedNormalFilePath = mergedCubeMapOutputPath / std::filesystem::path(mergedFileName + "_normal.exr");
	std::filesystem::path mergedDenoisedFilePath = mergedCubeMapOutputPath / std::filesystem::path(mergedFileName + "_denoised.exr");
	
	std::ostringstream mergeRendersCmd;
	mergeRendersCmd << "call ../mergeCubemap.bat";
	for(uint32_t i = 0; i < 6; ++i)
		mergeRendersCmd << " " << renderFilePaths[i];
	mergeRendersCmd << " " << mergedRenderFilePath.string();
	std::system(mergeRendersCmd.str().c_str());

	std::ostringstream mergeAlbedosCmd;
	mergeAlbedosCmd << "call ../mergeCubemap.bat ";
	for(uint32_t i = 0; i < 6; ++i)
		mergeAlbedosCmd << " " << albedoFilePaths[i];
	mergeAlbedosCmd << " " << mergedAlbedoFilePath.string();
	std::system(mergeAlbedosCmd.str().c_str());
	
	std::ostringstream mergeNormalsCmd;
	mergeNormalsCmd << "call ../mergeCubemap.bat ";
	for(uint32_t i = 0; i < 6; ++i)
		mergeNormalsCmd << " " << normalFilePaths[i];
	mergeNormalsCmd << " " << mergedNormalFilePath.string();
	std::system(mergeNormalsCmd.str().c_str());

	const std::string defaultBloomFile = "../../media/kernels/physical_flare_512.exr";
	const std::string defaultTonemapperArgs = "ACES=0.4,0.8";
	constexpr auto defaultBloomScale = 0.1f;
	constexpr auto defaultBloomIntensity = 0.1f;
	auto bloomFilePathStr = (denoiserArgs.bloomFilePath.string().empty()) ? defaultBloomFile : denoiserArgs.bloomFilePath.string();
	auto bloomScale = (denoiserArgs.bloomScale == 0.0f) ? defaultBloomScale : denoiserArgs.bloomScale;
	auto bloomIntensity = (denoiserArgs.bloomIntensity == 0.0f) ? defaultBloomIntensity : denoiserArgs.bloomIntensity;
	auto tonemapperArgs = (denoiserArgs.tonemapperArgs.empty()) ? defaultTonemapperArgs : denoiserArgs.tonemapperArgs;
	
	std::ostringstream denoiserCmd;
	// 1.ColorFile 2.AlbedoFile 3.NormalFile 4.BloomPsfFilePath(STRING) 5.BloomScale(FLOAT) 6.BloomIntensity(FLOAT) 7.TonemapperArgs(STRING)
	denoiserCmd << "call ../denoiser_hook.bat";
	denoiserCmd << " \"" << mergedRenderFilePath.string() << "\"";
	denoiserCmd << " \"" << mergedAlbedoFilePath.string() << "\"";
	denoiserCmd << " \"" << mergedNormalFilePath.string() << "\"";
	denoiserCmd << " \"" << bloomFilePathStr << "\"";
	denoiserCmd << " " << bloomScale;
	denoiserCmd << " " << bloomIntensity;
	denoiserCmd << " " << "\"" << tonemapperArgs << "\"";
	// NOTE/TODO/FIXME : Do as I say, not as I do
	// https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152177
	std::cout << "\n---[DENOISER_BEGIN]---" << std::endl;
	std::system(denoiserCmd.str().c_str());
	std::cout << "\n---[DENOISER_END]---" << std::endl;
	
	auto extractCubemapFaces = [&](const std::string& extension) -> void
	{
		std::ostringstream extractImagesCmd;
		auto mergedDenoisedWithoutExtension = std::filesystem::path(mergedDenoisedFilePath).replace_extension().string();
		extractImagesCmd << "call ../extractCubemap.bat ";
		extractImagesCmd << " " << std::to_string(cropOffsetX);
		extractImagesCmd << " " << std::to_string(cropOffsetY);
		extractImagesCmd << " " << mergedDenoisedWithoutExtension + extension;
		extractImagesCmd << " " << mergedDenoisedWithoutExtension + "_stripe" + extension;
		std::system(extractImagesCmd.str().c_str());
	};

	extractCubemapFaces(".exr");
	extractCubemapFaces(".png");
	extractCubemapFaces(".jpg");
}

// one day it will just work like that
//#include <nbl/builtin/glsl/sampling/box_muller_transform.glsl>

bool Renderer::render(nbl::ITimer* timer, const float kappa, const float Emin, const bool transformNormals, const bool beauty)
{
	if (m_cullPushConstants.maxGlobalInstanceCount==0u)
		return true;

	auto camera = m_smgr->getActiveCamera();
	camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(timer->getTime()).count());
	camera->render();

	// check if camera moved
	{
		auto properEquals = [](const core::matrix4x3& lhs, const core::matrix4x3& rhs) -> bool
		{
			const float rotationTolerance = 1.01f;
			const float positionTolerance = 1.005f;
			for (auto r=0; r<3u; r++)
			for (auto c=0; c<4u; c++)
			{
				const float ratio = core::abs((&rhs.getColumn(c).X)[r]/(&lhs.getColumn(c).X)[r]);
				// TODO: do by ULP
				if (core::isnan(ratio) || core::isinf(ratio))
					continue;
				const float tolerance = c!=3u ? rotationTolerance:positionTolerance;
				if (ratio>tolerance || ratio*tolerance<1.f)
					return false;
			}
			return true;
		};
		auto tform = camera->getRelativeTransformationMatrix();
		if (!properEquals(tform,m_prevCamTform))
		{
			m_framesDispatched = 0u;		
			m_prevView = camera->getViewMatrix();
			m_prevCamTform = tform;
		}
		else // need this to stop mouse cursor drift
			camera->setRelativeTransformationMatrix(m_prevCamTform);
	}

	// TODO: update positions and rr->Commit() if stuff starts to move

	if(compileShadersFuture.valid())
	{
		bool compiledShaders = compileShadersFuture.get();
		if(compiledShaders)
		{
			m_cullPipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_cullPipelineLayout), core::smart_refctd_ptr(m_cullGPUShader));	
			{
				IGPUSpecializedShader* shaders[] = {m_vertGPUShader.get(),m_fragGPUShader.get()};
				SPrimitiveAssemblyParams primitiveAssembly;
				primitiveAssembly.primitiveType = EPT_TRIANGLE_LIST;
				SRasterizationParams raster;
				raster.faceCullingMode = EFCM_NONE;
				auto _visibilityBufferFillPipelineLayout = m_driver->createGPUPipelineLayout(
					nullptr,nullptr,
					core::smart_refctd_ptr(m_rasterInstanceDataDSLayout),
					core::smart_refctd_ptr(m_additionalGlobalDSLayout),
					core::smart_refctd_ptr(m_cullDSLayout)
				);
				m_visibilityBufferFillPipeline = m_driver->createGPURenderpassIndependentPipeline(
					nullptr,std::move(_visibilityBufferFillPipelineLayout),shaders,shaders+2u,
					SVertexInputParams{},SBlendParams{},primitiveAssembly,raster
				);
			}
			m_raygenPipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_raygenPipelineLayout), core::smart_refctd_ptr(m_raygenGPUShader));
			m_closestHitPipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_closestHitPipelineLayout), core::smart_refctd_ptr(m_closestHitGPUShader));
			m_resolvePipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_resolvePipelineLayout), core::smart_refctd_ptr(m_resolveGPUShader));
			bool createPipelinesSuceess = m_cullPipeline && m_raygenPipeline && m_closestHitPipeline && m_resolvePipeline;
			if(!createPipelinesSuceess)
				std::cout << "Pipeline Compilation Failed." << std::endl;
		}
		else
			std::cout << "Shader Compilation Failed." << std::endl;
	}

	// only advance frame if rendering a beauty
	if (beauty)
		m_framesDispatched++;

	// raster jittered frame
	{
		const auto projMat = camera->getProjectionMatrix();
		const bool isOrtho = (projMat.rows[3]==core::vectorSIMDf(0.f,0.f,0.f,1.f)).all();
		// jitter with AA AntiAliasingSequence
		const auto modifiedProj = [&](uint32_t frameID)
		{
			const float stddev = 0.5f;
			const float* sample = AntiAliasingSequence[frameID%AntiAliasingSequenceLength];
			const float phi = core::PI<float>()*(2.f*sample[1]-1.f);
			const float sinPhi = sinf(phi);
			const float cosPhi = cosf(phi);
			const float truncated = sample[0]*0.99999f+0.00001f;
			const float r = sqrtf(-2.f*logf(truncated))*stddev;
			core::matrix4SIMD jitterMatrix;
			jitterMatrix.rows[0][3] = cosPhi*r*m_rcpPixelSize.x;
			jitterMatrix.rows[1][3] = sinPhi*r*m_rcpPixelSize.y;
			return core::concatenateBFollowedByA(jitterMatrix,projMat);
		}(m_framesDispatched);
		m_raytraceCommonData.rcpFramesDispatched = 1.f/float(m_framesDispatched);
		m_raytraceCommonData.textureFootprintFactor = core::inversesqrt(core::min<float>(m_framesDispatched ? m_framesDispatched:1u,Renderer::AntiAliasingSequenceLength));
		
		// work out the inverse of the Rotation component of the View applied before Projection
		core::matrix4SIMD viewDirReconFactorsT;
		{
			core::matrix4SIMD viewRotProjInvT;
			{
				core::matrix4SIMD viewRotProj(m_prevView);
				viewRotProj.setTranslation(core::vectorSIMDf(0.f));
				if (!core::concatenateBFollowedByA(modifiedProj,viewRotProj).getInverseTransform<core::matrix4SIMD::E_MATRIX_INVERSE_PRECISION::EMIP_64BBIT>(viewRotProjInvT))
					std::cout << "Couldn't calculate viewProjection matrix's inverse. something is wrong." << std::endl;
				viewRotProjInvT = core::transpose(viewRotProjInvT);
			}
			if (isOrtho) // normalizedV = -viewRotProjInv
			{
				viewDirReconFactorsT.rows[0].set(0,0,0,0);
				viewDirReconFactorsT.rows[1].set(0,0,0,0);
				viewDirReconFactorsT.rows[2] = -viewRotProjInvT.rows[2];
			}
			else
			{
				// note that we don't care about W coordinate & last row, W-divide, etc. because later normalization murders any scale on the vector
				// normalizedV = normalize(-viewRotProjInv*vec3(NDC*vec2(0.5,-0.5)+vec2(-0.5,0.5),1))
				viewDirReconFactorsT.rows[0] = viewRotProjInvT.rows[0]*(-2.f);
				viewDirReconFactorsT.rows[1] = viewRotProjInvT.rows[1]*(+2.f);
				viewDirReconFactorsT.rows[2] = viewRotProjInvT.rows[0]-viewRotProjInvT.rows[1]-viewRotProjInvT.rows[2]-viewRotProjInvT.rows[3];
			}
			}
		
		// cull batches
		m_driver->bindComputePipeline(m_cullPipeline.get());
		{
			const auto* _cullPipelineLayout = m_cullPipeline->getLayout();

			IGPUDescriptorSet* descriptorSets[] = { m_globalBackendDataDS.get(),m_cullDS.get() };
			m_driver->bindDescriptorSets(EPBP_COMPUTE,_cullPipelineLayout,0u,2u,descriptorSets,nullptr);
			
			m_cullPushConstants.viewProjMatrix = core::concatenateBFollowedByA(modifiedProj,m_prevView);
			m_cullPushConstants.viewProjDeterminant = core::determinant(m_cullPushConstants.viewProjMatrix);
			m_driver->pushConstants(_cullPipelineLayout,ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t),&m_cullPushConstants);
		}
		// TODO: Occlusion Culling against HiZ Buffer
		m_driver->dispatch(m_cullWorkGroups, 1u, 1u);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_COMMAND_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);

		m_driver->setRenderTarget(m_visibilityBuffer);
		{ // clear
			m_driver->clearZBuffer();
			uint32_t clearTriangleID[4] = {0xffffffffu,0,0,0};
			m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0, clearTriangleID);
		}
		// all batches draw with the same pipeline
		m_driver->bindGraphicsPipeline(m_visibilityBufferFillPipeline.get());
		{
			IGPUDescriptorSet* descriptorSets[] = { m_rasterInstanceDataDS.get(),m_additionalGlobalDS.get(),m_cullDS.get() };
			m_driver->bindDescriptorSets(EPBP_GRAPHICS,m_visibilityBufferFillPipeline->getLayout(),0u,3u,descriptorSets,nullptr);
		}
		for (const auto& call : m_mdiDrawCalls)
		{
			const asset::SBufferBinding<IGPUBuffer> nullBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {};
			m_driver->drawIndexedIndirect(
				nullBindings,EPT_TRIANGLE_LIST,EIT_16BIT,m_indexBuffer.get(),
				m_indirectDrawBuffers[m_cullPushConstants.currentCommandBufferIx].get(),
				call.mdiOffset*sizeof(DrawElementsIndirectCommand_t),call.mdiCount,sizeof(DrawElementsIndirectCommand_t)
			);
		}
		// flip MDI buffers
		m_cullPushConstants.currentCommandBufferIx ^= 0x01u;

		// prepare camera data for raytracing
		viewDirReconFactorsT.rows[3] = core::vectorSIMDf().set(camera->getAbsolutePosition());
		m_raytraceCommonData.viewDirReconFactors = core::transpose(viewDirReconFactorsT).extractSub3x4();
	}
	// raygen
	{
		// vertex 0 is camera
		m_raytraceCommonData.setPathDepth(beauty ? 0u:(~0u));

		//
		video::IGPUDescriptorSet* sameDS[2] = {m_raygenDS.get(),m_raygenDS.get()};
		preDispatch(m_raygenPipeline->getLayout(),sameDS);

		//
		m_driver->bindComputePipeline(m_raygenPipeline.get());
		m_driver->dispatch(m_raygenWorkGroups[0],m_raygenWorkGroups[1],1);
	}
	// path trace
	if (beauty)
	{
		while (m_raytraceCommonData.getPathDepth()!=m_staticViewData.maxPathDepth)
		{
			uint32_t raycount;
			 if(!traceBounce(raycount))
				 return false;
			 if (raycount==0u)
				 break;
		}
	}
	// mostly writes to accumulation buffers and SSBO clears
	// probably wise to flush all caches (in the future can optimize to texture_fetch|shader_image_access|shader_storage_buffer|blit|texture_download|...)
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);

	// resolve pseudo-MSAA
	if (beauty)
	{
		m_raytraceCommonData.frameLowDiscrepancySequenceShift = (m_raytraceCommonData.frameLowDiscrepancySequenceShift+getSamplesPerPixelPerDispatch())%maxSensorSamples;

		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_resolvePipeline->getLayout(),0u,1u,&m_resolveDS.get(),nullptr);
		m_driver->bindComputePipeline(m_resolvePipeline.get());
		if (transformNormals)
			m_driver->pushConstants(m_resolvePipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(m_prevView),&m_prevView);
		else
		{
			decltype(m_prevView) identity;
			m_driver->pushConstants(m_resolvePipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(identity),&identity);
		}
		{
			const auto reweightingParams = nbl_glsl_RWMC_computeReweightingParameters(
				m_staticViewData.cascadeParams.penultimateCascadeIx+2u,
				m_staticViewData.cascadeParams.base,
				m_framesDispatched*m_staticViewData.samplesPerPixelPerDispatch,
				Emin,kappa
			);
			m_driver->pushConstants(m_resolvePipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,sizeof(m_prevView),sizeof(reweightingParams),&reweightingParams);
		}
		m_driver->dispatch(m_raygenWorkGroups[0],m_raygenWorkGroups[1],1);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
			// because of direct to screen resolve
			|GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT
		);
	}

	// TODO: autoexpose properly
	return true;
}

void Renderer::preDispatch(const video::IGPUPipelineLayout* pipelineLayout, video::IGPUDescriptorSet*const *const lastDS)
{
	// increment depth
	const auto depth = m_raytraceCommonData.getPathDepth()+1;
	m_raytraceCommonData.setPathDepth(depth);
	const uint32_t descSetIx = depth&0x1u;
	m_driver->pushConstants(pipelineLayout,ISpecializedShader::ESS_COMPUTE,0u,sizeof(RaytraceShaderCommonData_t),&m_raytraceCommonData);

	// advance rayCountWriteIx
	m_raytraceCommonData.advanceWriteIndex();
	
	IGPUDescriptorSet* descriptorSets[4] = {m_globalBackendDataDS.get(),m_additionalGlobalDS.get(),m_commonRaytracingDS[descSetIx].get(),lastDS[descSetIx]};
	m_driver->bindDescriptorSets(EPBP_COMPUTE,pipelineLayout,0u,4u,descriptorSets,nullptr);
}

bool Renderer::traceBounce(uint32_t& raycount)
{
	// probably wise to flush all caches (in the future can optimize to texture_fetch|shader_image_access|shader_storage_buffer|blit|texture_download|...)
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);
	m_driver->copyBuffer(m_rayCountBuffer.get(),m_littleDownloadBuffer.get(),sizeof(uint32_t)*m_raytraceCommonData.getReadIndex(),0u,sizeof(uint32_t));
	glFinish(); // sync CPU to GL
	raycount = *reinterpret_cast<uint32_t*>(m_littleDownloadBuffer->getBoundMemory()->getMappedPointer());

	if (raycount)
	{
		// trace rays
		m_totalRaysCast += raycount;
		{
			const uint32_t descSetIx = m_raytraceCommonData.getPathDepth()&0x1u;

			auto commandQueue = m_rrManager->getCLCommandQueue();
			const cl_mem clObjects[] = {m_rayBuffer[descSetIx].asRRBuffer.second,m_intersectionBuffer[descSetIx].asRRBuffer.second};
			const auto objCount = sizeof(clObjects)/sizeof(cl_mem);
			cl_event acquired=nullptr, raycastDone=nullptr;
			// run the raytrace queries
			{
				ocl::COpenCLHandler::ocl.pclEnqueueAcquireGLObjects(commandQueue,objCount,clObjects,0u,nullptr,&acquired);

				clEnqueueWaitForEvents(commandQueue,1u,&acquired);
				m_rrManager->getRadeonRaysAPI()->QueryIntersection(
					m_rayBuffer[descSetIx].asRRBuffer.first,raycount,
					m_intersectionBuffer[descSetIx].asRRBuffer.first,nullptr,nullptr
				);
				clEnqueueMarker(commandQueue,&raycastDone);
			}

			// sync CPU to CL
			cl_event released;
			ocl::COpenCLHandler::ocl.pclEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, &released);
			ocl::COpenCLHandler::ocl.pclFlush(commandQueue);

			cl_int retval = -1;
			auto startWait = std::chrono::steady_clock::now();
			constexpr auto timeoutInSeconds = 20ull;
			bool timedOut = false;
			do {
				const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-startWait).count();
				if (elapsed > timeoutInSeconds * 1'000'000ull)
				{
					timedOut = true;
					break;
				}

				std::this_thread::yield();
				ocl::COpenCLHandler::ocl.pclGetEventInfo(released, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &retval, nullptr);
			} while(retval != CL_COMPLETE);
		
			if(timedOut)
			{
				std::cout << "[ERROR] RadeonRays Timed Out" << std::endl;
				return false;
			}

			if (static_cast<COpenGLDriver*>(m_driver)->runningInRenderdoc())
			{
				auto touchAllBytes = [](IGPUBuffer* buf)->void
				{
					auto ptr = reinterpret_cast<uint8_t*>(buf->getBoundMemory()->getMappedPointer());
				};
				touchAllBytes(m_intersectionBuffer[descSetIx].buffer.get());
			}
		}

	
		// compute bounce (accumulate contributions and optionally generate rays)
		{
			preDispatch(m_closestHitPipeline->getLayout(),&m_closestHitDS->get());

			m_driver->bindComputePipeline(m_closestHitPipeline.get());
			m_driver->dispatch((raycount-1u)/WORKGROUP_SIZE+1u,1u,1u);
		}
	}

	return true;
}