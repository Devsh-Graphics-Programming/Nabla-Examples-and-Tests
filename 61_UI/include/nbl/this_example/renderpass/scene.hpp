#ifndef __NBL_THIS_EXAMPLE_SCENE_H_INCLUDED__
#define __NBL_THIS_EXAMPLE_SCENE_H_INCLUDED__

class CScene
{
public:
	CScene(const IGeometryCreator* geometry)
	{
		struct
		{
			const IGeometryCreator* gc;

			const std::vector<OBJECT_CPU> basic =
			{
				{.meta = {.type = EOT_CUBE, .name = "Cube Mesh" }, .data = gc->createCubeMesh(vector3df(1.f, 1.f, 1.f)) },
				{.meta = {.type = EOT_SPHERE, .name = "Sphere Mesh" }, .data = gc->createSphereMesh(2, 16, 16) },
				{.meta = {.type = EOT_CYLINDER, .name = "Cylinder Mesh" }, .data = gc->createCylinderMesh(2, 2, 20) },
				{.meta = {.type = EOT_RECTANGLE, .name = "Rectangle Mesh" }, .data = gc->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3)) },
				{.meta = {.type = EOT_DISK, .name = "Disk Mesh" }, .data = gc->createDiskMesh(2, 30) },
				{.meta = {.type = EOT_ARROW, .name = "Arrow Mesh" }, .data = gc->createArrowMesh() },
			}, cone =
			{
				{.meta = {.type = EOT_CONE, .name = "Cone Mesh" }, .data = gc->createConeMesh(2, 3, 10) }
			}, ico =
			{
				{.meta = {.type = EOT_ICOSPHERE, .name = "Icoshpere Mesh" }, .data = gc->createIcoSphere(1, 3, true) }
			};
		} geometries { .gc = geometry };

		auto createBundleGPUData = [&]<nbl::core::StringLiteral vPath, nbl::core::StringLiteral fPath>(const std::vector<OBJECT_CPU>& objects) -> void
		{
			SHADERS_GPU shaders;
			{
				struct
				{
					const nbl::system::SBuiltinFile vertex = ::geometry::creator::spirv::builtin::get_resource<vPath>();
					const nbl::system::SBuiltinFile fragment = ::geometry::creator::spirv::builtin::get_resource<fPath>();
				} spirv;

				auto createShader = [&](const nbl::system::SBuiltinFile& in, asset::IShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<video::IGPUShader>
				{
					const auto buffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(in.size, (void*)in.contents, core::adopt_memory);
					const auto shader = make_smart_refctd_ptr<ICPUShader>(nbl::core::smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, "");

					return m_device->createShader(shader.get()); // also first should look for cached/already created to not duplicate
				};

				shaders.vertex = createShader(spirv.vertex, IShader::E_SHADER_STAGE::ESS_VERTEX);
				shaders.fragment = createShader(spirv.fragment, IShader::E_SHADER_STAGE::ESS_FRAGMENT);
			}

			for (const auto& inObject : objects)
			{
				auto& outObject = referenceObjects.emplace_back();

				if (!createGPUData<vPath, fPath>(inObject, outObject, shaders, pipelineLayout.get(), renderpass))
					return logFail("Could not create pass data!");
			}
		};

		createBundleGPUData.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.vertex.spv"), NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (geometries.basic);
		createBundleGPUData.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.cone.vertex.spv"), NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (geometries.cone);		// note we reuse basic fragment shader
		createBundleGPUData.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.ico.vertex.spv"), NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("geometryCreator/spirv/gc.basic.fragment.spv") > (geometries.ico);		// note we reuse basic fragment shader
	}
	~CScene() {}
private:

	enum E_OBJECT_TYPES : uint8_t
	{
		EOT_CUBE,
		EOT_SPHERE,
		EOT_CYLINDER,
		EOT_RECTANGLE,
		EOT_DISK,
		EOT_ARROW,
		EOT_CONE,
		EOT_ICOSPHERE
	};

	struct OBJECT_META
	{
		E_OBJECT_TYPES type;
		std::string_view name;
	};

	struct OBJECT_GPU
	{
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
		core::smart_refctd_ptr<video::IGPUBuffer> vertexBuffer, indexBuffer;
		E_INDEX_TYPE indexType;
		uint32_t indexCount;
	};

	struct OBJECT_CPU
	{
		OBJECT_META meta;
		CGeometryCreator::return_type data;
	};

	struct SHADERS_GPU
	{
		core::smart_refctd_ptr<video::IGPUShader> vertex, geometry, fragment;
	};

	using OBJECT_DATA = std::pair<OBJECT_GPU, OBJECT_META>;

	std::vector<OBJECT_DATA> referenceObjects; // all possible objects, lets use geometry creator

	template<nbl::core::StringLiteral vPath, nbl::core::StringLiteral fPath>
	bool createGPUData(const OBJECT_CPU& inData, OBJECT_DATA& outData, const SHADERS_GPU& shaders, const video::IGPUPipelineLayout* pl, const video::IGPURenderpass* rp)
	{
		// meta
		outData.second.name = inData.meta.name;
		outData.second.type = inData.meta.type;

		SBlendParams blendParams{};
		{
			blendParams.logicOp = ELO_NO_OP;

			auto& param = blendParams.blendParams[0];
			param.srcColorFactor = EBF_SRC_ALPHA;//VK_BLEND_FACTOR_SRC_ALPHA;
			param.dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			param.colorBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
			param.srcAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			param.dstAlphaFactor = EBF_ZERO;//VK_BLEND_FACTOR_ZERO;
			param.alphaBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
			param.colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);//VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		}

		SRasterizationParams rasterizationParams{};
		rasterizationParams.faceCullingMode = EFCM_NONE;
		{
			const IGPUShader::SSpecInfo specs[] =
			{
				{.entryPoint = "VSMain", .shader = shaders.vertex.get() },
				{.entryPoint = "PSMain", .shader = shaders.fragment.get() }
			};

			IGPUGraphicsPipeline::SCreationParams params[1];
			{
				auto& param = params[0];
				param.layout = pl;
				param.shaders = specs;
				param.renderpass = rp;
				param.cached = { .vertexInput = inData.data.inputParams, .primitiveAssembly = inData.data.assemblyParams, .rasterization = rasterizationParams, .blend = blendParams, .subpassIx = 0u };
			};

			outData.first.indexCount = geo.indexCount;
			outData.first.indexType = geo.indexType;

			// first should look for cached pipeline to not duplicate but lets leave how it is now
			if (!m_device->createGraphicsPipelines(nullptr, params, &outData.first.pipeline))
				return false;

			if (!createVIBuffers(hook, geo))
				return false;

			return true;
		}
	}

	bool createVIBuffers(const OBJECT_CPU& inData, OBJECT_DATA& outData)
	{
		const auto mask = m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

		auto vBuffer = core::smart_refctd_ptr(inData.data.bindings[0].buffer); // no offset
		auto iBuffer = core::smart_refctd_ptr(inData.data.indexBuffer.buffer); // no offset

		outData.first.vertexBuffer = m_device->createBuffer({ {.size = vBuffer->getSize(), .usage = core::bitflag(asset::IBuffer::EUF_VERTEX_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} });
		outData.first.indexBuffer = iBuffer ? m_device->createBuffer({ {.size = iBuffer->getSize(), .usage = core::bitflag(asset::IBuffer::EUF_INDEX_BUFFER_BIT) | asset::IBuffer::EUF_VERTEX_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF} }) : nullptr;

		if (!outData.first.vertexBuffer)
			return false;

		if (oData.indexType != EIT_UNKNOWN)
			if (!outData.first.indexBuffer)
				return false;

		for (auto it : { outData.first.vertexBuffer , outData.first.indexBuffer })
		{
			if (it)
			{
				IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = it->getMemoryReqs();
				reqs.memoryTypeBits &= mask;

				m_device->allocate(reqs, it.get());
			}
		}

		{
			auto fillGPUBuffer = [&m_logger = m_logger](smart_refctd_ptr<ICPUBuffer> cBuffer, smart_refctd_ptr<IGPUBuffer> gBuffer)
			{
				auto binding = gBuffer->getBoundMemory();

				if (!binding.memory->map({ 0ull, binding.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
				{
					m_logger->log("Could not map device memory", system::ILogger::ELL_ERROR);
					return false;
				}

				if (!binding.memory->isCurrentlyMapped())
				{
					m_logger->log("Buffer memory is not mapped!", system::ILogger::ELL_ERROR);
					return false;
				}

				auto* mPointer = binding.memory->getMappedPointer();
				memcpy(mPointer, cBuffer->getPointer(), gBuffer->getSize());
				binding.memory->unmap();

				return true;
			};

			if (!fillGPUBuffer(vBuffer, outData.first.vertexBuffer))
				return false;

			if (outData.first.indexBuffer)
				if (!fillGPUBuffer(iBuffer, outData.first.indexBuffer))
					return false;
		}

		return true;
	}
};

#endif // __NBL_THIS_EXAMPLE_SCENE_H_INCLUDED__