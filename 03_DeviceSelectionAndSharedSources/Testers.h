// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_TESTERS_H_INCLUDED_
#define _NBL_TESTERS_H_INCLUDED_

#include "nbl/examples/examples.hpp"

using namespace nbl;

class IntrospectionTesterBase
{

public:
	IntrospectionTesterBase(const std::string& functionToTestName)
		: m_functionToTestName(functionToTestName) {};

	void virtual performTests(video::IPhysicalDevice* physicalDevice, video::ILogicalDevice* device, system::ILogger* logger, asset::IAssetManager* assetMgr) = 0;

	virtual ~IntrospectionTesterBase() {};

protected:
	const std::string m_functionToTestName = "";

protected:
	static std::pair<smart_refctd_ptr<IShader>, smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData>> compileHLSLShaderAndTestIntrospection(
		video::IPhysicalDevice* physicalDevice, video::ILogicalDevice* device, system::ILogger* logger, asset::IAssetManager* assetMgr, const std::string& shaderPath, CSPIRVIntrospector& introspector)
	{
		IAssetLoader::SAssetLoadParams lp = {};
		lp.logger = logger;
		lp.workingDirectory = ""; // virtual root
		// this time we load a shader directly from a file
		auto assetBundle = assetMgr->getAsset(shaderPath, lp);
		const auto assets = assetBundle.getContents();
		const auto* metadata = assetBundle.getMetadata();
		if (assets.empty() || assetBundle.getAssetType() != IAsset::ET_SHADER)
		{
			logFail(logger, "Could not load shader!");
			assert(0);
		}
		const auto hlslMetadata = static_cast<const CHLSLMetadata*>(metadata);
		const auto shaderStage = hlslMetadata->shaderStages->front();

		// It would be super weird if loading a shader from a file produced more than 1 asset
		assert(assets.size() == 1);
		smart_refctd_ptr<IShader> source = IAsset::castDown<IShader>(assets[0]);

		smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData> introspection;
		{
			// The Asset Manager has a Default Compiler Set which contains all built-in compilers (so it can try them all)
			auto* compilerSet = assetMgr->getCompilerSet();

			// This time we use a more "generic" option struct which works with all compilers
			nbl::asset::IShaderCompiler::SCompilerOptions options = {};
			// The Shader Asset Loaders deduce the stage from the file extension,
			// if the extension is generic (.glsl or .hlsl) the stage is unknown.
			// But it can still be overriden from within the source with a `#pragma shader_stage` 
			options.stage = shaderStage == IShader::E_SHADER_STAGE::ESS_COMPUTE ? shaderStage : IShader::E_SHADER_STAGE::ESS_VERTEX; // TODO: do smth with it
			options.preprocessorOptions.targetSpirvVersion = device->getPhysicalDevice()->getLimits().spirvVersion;
			// we need to perform an unoptimized compilation with source debug info or we'll lose names of variable sin the introspection
			options.spirvOptimizer = nullptr;
			options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
			// The nice thing is that when you load a shader from file, it has a correctly set `filePathHint`
			// so it plays nicely with the preprocessor, and finds `#include`s without intervention.
			options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
			options.preprocessorOptions.logger = logger;
			options.preprocessorOptions.includeFinder = compilerSet->getShaderCompiler(source->getContentType())->getDefaultIncludeFinder();

			auto spirvUnspecialized = compilerSet->compileToSPIRV(source.get(), options);
			const CSPIRVIntrospector::CStageIntrospectionData::SParams inspctParams = { .entryPoint = "main", .shader = spirvUnspecialized };

			introspection = introspector.introspect(inspctParams);
			if (!introspection)
			{
				logFail(logger, "SPIR-V Introspection failed, probably the required SPIR-V compilation failed first!");
				return std::pair(nullptr, nullptr);
			}

			{
				auto* srcContent = spirvUnspecialized->getContent();

				system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
				physicalDevice->getSystem()->createFile(future, system::path("../app_resources/compiled.spv"), system::IFileBase::ECF_WRITE);
				if (auto file = future.acquire(); file && bool(*file))
				{
					system::IFile::success_t succ;
					(*file)->write(succ, srcContent->getPointer(), 0, srcContent->getSize());
					succ.getBytesProcessed(true);
				}
			}

			// now we need to swap out the HLSL for SPIR-V
			source = std::move(spirvUnspecialized);
		}

		return std::pair(source, introspection);
	}

	void confirmExpectedOutput(system::ILogger* logger, bool value, bool expectedValue)
	{
		if (value != expectedValue)
		{
			logger->log("\"CSPIRVIntrospector::CPipelineIntrospectionData::merge\" function FAIL, incorrect output.",
				ILogger::E_LOG_LEVEL::ELL_ERROR);
		}
		else
		{
			logger->log("\"CSPIRVIntrospector::CPipelineIntrospectionData::merge\" function SUCCESS, correct output.",
				ILogger::E_LOG_LEVEL::ELL_PERFORMANCE);
		}
	}

	template<typename... Args>
	static inline bool logFail(system::ILogger* logger, const char* msg, Args&&... args)
	{
		logger->log(msg, system::ILogger::ELL_ERROR, std::forward<Args>(args)...);
		return false;
	}
};

class MergeTester final : public IntrospectionTesterBase
{
public:
	MergeTester(const std::string& functionToTestName)
		: IntrospectionTesterBase(functionToTestName) {};

	void virtual performTests(video::IPhysicalDevice* physicalDevice, video::ILogicalDevice* device, system::ILogger* logger, asset::IAssetManager* assetMgr)
	{
		constexpr std::array mergeTestShadersPaths = {
				"app_resources/pplnLayoutMergeTest/shader_0.comp.hlsl",
				"app_resources/pplnLayoutMergeTest/shader_1.comp.hlsl",
				"app_resources/pplnLayoutMergeTest/shader_2.comp.hlsl",
				"app_resources/pplnLayoutMergeTest/shader_3.comp.hlsl",
				"app_resources/pplnLayoutMergeTest/shader_4.comp.hlsl",
				"app_resources/pplnLayoutMergeTest/shader_5.comp.hlsl"
		};
		constexpr uint32_t MERGE_TEST_SHADERS_CNT = mergeTestShadersPaths.size();

		CSPIRVIntrospector introspector[MERGE_TEST_SHADERS_CNT];
		smart_refctd_ptr<const CSPIRVIntrospector::CStageIntrospectionData> introspections[MERGE_TEST_SHADERS_CNT];

		for (uint32_t i = 0u; i < MERGE_TEST_SHADERS_CNT; ++i)
		{
			auto sourceIntrospectionPair = compileHLSLShaderAndTestIntrospection(physicalDevice, device, logger, assetMgr, mergeTestShadersPaths[i], introspector[i]);
			introspections[i] = sourceIntrospectionPair.second;
		}

		core::smart_refctd_ptr<CSPIRVIntrospector::CPipelineIntrospectionData> pplnIntroData;
		pplnIntroData = core::make_smart_refctd_ptr<CSPIRVIntrospector::CPipelineIntrospectionData>();

		// should merge successfully since shader is not messed up and it is the first merge
		confirmExpectedOutput(logger, pplnIntroData->merge(introspections[0].get()), true);
		// should merge successfully since pipeline layout of "shader_1.comp.hlsl" is compatible with "shader_0.comp.hlsl"
		confirmExpectedOutput(logger, pplnIntroData->merge(introspections[1].get()), true);
		// should merge since pipeline layout of "shader_2.comp.hlsl" is not compatible with "shader_0.comp.hlsl"
		confirmExpectedOutput(logger, pplnIntroData->merge(introspections[2].get()), true);

		pplnIntroData = core::make_smart_refctd_ptr<CSPIRVIntrospector::CPipelineIntrospectionData>();

		// should not merge since run-time sized destriptor of "shader_3.comp.hlsl" is not last
		confirmExpectedOutput(logger, pplnIntroData->merge(introspections[3].get()), false);

		pplnIntroData = core::make_smart_refctd_ptr<CSPIRVIntrospector::CPipelineIntrospectionData>();

		// should merge successfully since shader is not messed up and it is the first merge
		confirmExpectedOutput(logger, pplnIntroData->merge(introspections[4].get()), true);
		// TODO: should merge successfully since shader 5 is compatible with shader 4, it is allowed for last binding in one shader to be run-time sized and statically sized in the other
		confirmExpectedOutput(logger, pplnIntroData->merge(introspections[5].get()), true);
	}
};

class PredefinedLayoutTester final : public IntrospectionTesterBase
{
public:
	PredefinedLayoutTester(const std::string& functionToTestName)
		: IntrospectionTesterBase(functionToTestName) {};

	void virtual performTests(video::IPhysicalDevice* physicalDevice, video::ILogicalDevice* device, system::ILogger* logger, asset::IAssetManager* assetMgr)
	{
		constexpr std::array mergeTestShadersPaths = {
				"app_resources/pplnLayoutCreationWithPredefinedLayoutTest/shader_0.comp.hlsl",
				"app_resources/pplnLayoutCreationWithPredefinedLayoutTest/shader_1.comp.hlsl",
				"app_resources/pplnLayoutCreationWithPredefinedLayoutTest/shader_2.comp.hlsl",
				"app_resources/pplnLayoutCreationWithPredefinedLayoutTest/shader_3.comp.hlsl",
				"app_resources/pplnLayoutCreationWithPredefinedLayoutTest/shader_4.comp.hlsl",
				"app_resources/pplnLayoutCreationWithPredefinedLayoutTest/shader_5.comp.hlsl"
		};
		constexpr uint32_t MERGE_TEST_SHADERS_CNT = mergeTestShadersPaths.size();

		CSPIRVIntrospector introspector[MERGE_TEST_SHADERS_CNT];
		smart_refctd_ptr<IShader> sources[MERGE_TEST_SHADERS_CNT];

		for (uint32_t i = 0u; i < MERGE_TEST_SHADERS_CNT; ++i)
		{
			auto sourceIntrospectionPair = compileHLSLShaderAndTestIntrospection(physicalDevice, device, logger, assetMgr, mergeTestShadersPaths[i], introspector[i]);
			// TODO: disctinct functions for shader compilation and introspection
			sources[i] = sourceIntrospectionPair.first;
		}

		constexpr uint32_t BINDINGS_DS_0_CNT = 1u;
		const ICPUDescriptorSetLayout::SBinding bindingsDS0[BINDINGS_DS_0_CNT] = {
			{
				.binding = 0,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count = 1,
				.immutableSamplers = nullptr
			}
		};

		constexpr uint32_t BINDINGS_DS_1_CNT = 2u;
		const ICPUDescriptorSetLayout::SBinding bindingsDS1[BINDINGS_DS_1_CNT] = {
				{
					.binding = 0,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1,
					.immutableSamplers = nullptr
				},
				{
					.binding = 1,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = ICPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 2,
					.immutableSamplers = nullptr
				}
		};

		core::smart_refctd_ptr<ICPUDescriptorSetLayout> dsLayout0 = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindingsDS0, bindingsDS0 + BINDINGS_DS_0_CNT);
		core::smart_refctd_ptr<ICPUDescriptorSetLayout> dsLayout1 = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindingsDS1, bindingsDS1 + BINDINGS_DS_1_CNT);

		if (!dsLayout0 || !dsLayout1)
		{
			logFail(logger, "Failed to create a Descriptor Layout!\n");
			return;
		}

		SPushConstantRange pc;
		pc.offset = 0u;
		pc.size = 5 * sizeof(uint32_t);
		pc.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE;

		smart_refctd_ptr<ICPUPipelineLayout> predefinedPplnLayout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(std::span<const asset::SPushConstantRange>({ pc }), std::move(dsLayout0), std::move(dsLayout1), nullptr, nullptr);
		if (!predefinedPplnLayout)
		{
			logFail(logger, "Failed to create a Pipeline Layout!\n");
			return;
		}

		bool pplnCreationSuccess[MERGE_TEST_SHADERS_CNT];
		for (uint32_t i = 0u; i < MERGE_TEST_SHADERS_CNT; ++i)
		{
			ICPUPipelineBase::SShaderSpecInfo specInfo;
			specInfo.entryPoint = "main";
			specInfo.shader = sources[i];
			pplnCreationSuccess[i] = static_cast<bool>(introspector[i].createApproximateComputePipelineFromIntrospection(specInfo, core::smart_refctd_ptr<ICPUPipelineLayout>(predefinedPplnLayout)));
		}

		// DESCRIPTOR VALIDATION TESTS
	// layout from introspection is a subset of pre-defined layout, hence ppln creation should SUCCEED
		confirmExpectedOutput(logger, pplnCreationSuccess[0], true);
		// layout from introspection is NOT a subset (too many bindings in descriptor set 0) of pre-defined layout, hence ppln creation should FAIL
		confirmExpectedOutput(logger, pplnCreationSuccess[1], false);
		// layout from introspection is NOT a subset (pre-defined layout doesn't have descriptor set 2) of pre-defined layout, hence ppln creation should FAIL
		confirmExpectedOutput(logger, pplnCreationSuccess[2], false);
		// layout from introspection is NOT a subset (same bindings, different type of one of the bindings) of pre-defined layout, hence ppln creation should FAIL
		confirmExpectedOutput(logger, pplnCreationSuccess[3], false);

		// PUSH CONSTANTS VALIDATION TESTS
	// layout from introspection is a subset of pre-defined layout (Push constant size declared in shader are compatible), hence ppln creation should SUCCEED
		confirmExpectedOutput(logger, pplnCreationSuccess[4], true);
		// layout from introspection is NOT a subset of pre-defined layout (Push constant size declared in shader are NOT compatible), hence ppln creation should FAIL
		confirmExpectedOutput(logger, pplnCreationSuccess[5], false);
	}
};

class SandboxTester final : public IntrospectionTesterBase
{
public:
	SandboxTester(const std::string& functionToTestName)
		: IntrospectionTesterBase(functionToTestName) {};

	void virtual performTests(video::IPhysicalDevice* physicalDevice, video::ILogicalDevice* device, system::ILogger* logger, asset::IAssetManager* assetMgr)
	{
		CSPIRVIntrospector introspector;
		auto sourceIntrospectionPair = compileHLSLShaderAndTestIntrospection(physicalDevice, device, logger, assetMgr, "app_resources/test.hlsl", introspector);
		auto pplnIntroData = core::make_smart_refctd_ptr<CSPIRVIntrospector::CPipelineIntrospectionData>();
		confirmExpectedOutput(logger, pplnIntroData->merge(sourceIntrospectionPair.second.get()), true);

		sourceIntrospectionPair.second->debugPrint(logger);

		// TODO
		/*CSPIRVIntrospector introspector_test1;
		auto vtx_test1 = compileHLSLShaderAndTestIntrospection(physicalDevice, device, logger, assetMgr, "app_resources/vtx_test1.hlsl", introspector_test1);
		auto test1_frag = compileHLSLShaderAndTestIntrospection(physicalDevice, device, logger, assetMgr, "app_resources/frag_test1.hlsl", introspector_test1);

		CSPIRVIntrospector introspector_test2;
		auto test2_comp = compileHLSLShaderAndTestIntrospection(physicalDevice, device, logger, assetMgr, "app_resources/comp_test2_nestedStructs.hlsl", introspector_test2);

		CSPIRVIntrospector introspector_test3;
		auto test3_comp = compileHLSLShaderAndTestIntrospection(physicalDevice, device, logger, assetMgr, "app_resources/comp_test3_ArraysAndMatrices.hlsl", introspector_test3);

		CSPIRVIntrospector introspector_test4;
		auto test4_comp = compileHLSLShaderAndTestIntrospection(physicalDevice, device, logger, assetMgr, "app_resources/frag_test4_SamplersTexBuffAndImgStorage.hlsl", introspector_test4);*/
	}
};

#endif