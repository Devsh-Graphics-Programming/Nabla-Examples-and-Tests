# Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

if(NBL_BUILD_EXAMPLES)
	project(NablaExamples)

	if(NBL_BUILD_ANDROID)
		nbl_android_create_media_storage_apk()
	endif()

	#! Common api library & precompiled headers for Nabla framework examples
	add_subdirectory(common EXCLUDE_FROM_ALL)

	#! use "EXCLUDE_FROM_ALL" to exclude an example from the NablaExamples project
	#[[
		useful if we don't want the example to be tested by CI but still want
		the example's project to be generated

		https://cmake.org/cmake/help/latest/prop_tgt/EXCLUDE_FROM_ALL.html
	]]

	# showcase the use of `nbl::core`,`nbl::system` and `nbl::asset`
	add_subdirectory(01_HelloCoreSystemAsset)
	# showcase the use of `system::IApplicationFramework` and `nbl::video`
	add_subdirectory(02_HelloCompute)
	# showcase physical device selection, resource embedding and the use of identical headers in HLSL and C++
	add_subdirectory(03_DeviceSelectionAndSharedSources)
	# showcase the creation of windows and polling for input
	add_subdirectory(04_HelloUI)
	# showcase the semi-advanced use of Nabla's Streaming Buffers and BDA
	add_subdirectory(05_StreamingAndBufferDeviceAddressApp)
	# showcase the use of a graphics queue
	add_subdirectory(06_HelloGraphicsQueue)
	# showcase the set-up of multiple queues
	add_subdirectory(07_StagingAndMultipleQueues)
	# showcase the set-up of a swapchain and picking of a matching device
	add_subdirectory(08_HelloSwapchain)
	add_subdirectory(09_GeometryCreator)
  	# demonstrate the counting sort utility
	add_subdirectory(10_CountingSort)
	# showcase use of FFT for post-FX Bloom  effect
	add_subdirectory(11_FFT)
	#
	add_subdirectory(12_MeshLoaders)
	#
	add_subdirectory(13_MaterialCompilerTest EXCLUDE_FROM_ALL)

	# Waiting for a refactor
	#add_subdirectory(27_PLYSTLDemo)
	#add_subdirectory(33_Draw3DLine)

	# Unit Test Examples
	add_subdirectory(20_AllocatorTest)
	add_subdirectory(21_LRUCacheUnitTest)
	add_subdirectory(22_CppCompat)
	add_subdirectory(23_Arithmetic2UnitTest)
	add_subdirectory(24_ColorSpaceTest)
	add_subdirectory(25_FilterTest EXCLUDE_FROM_ALL)
	add_subdirectory(26_Blur)
	add_subdirectory(27_MPMCScheduler)	
	add_subdirectory(28_FFTBloom)
	add_subdirectory(29_Arithmetic2Bench)
	# add_subdirectory(36_CUDAInterop)

	# Showcase compute pathtracing
	add_subdirectory(30_ComputeShaderPathTracer)

	add_subdirectory(38_EXRSplit)
	# if (NBL_BUILD_MITSUBA_LOADER AND NBL_BUILD_OPTIX)
	#	add_subdirectory(39_DenoiserTonemapper)
	# endif()

	#add_subdirectory(43_SumAndCDFFilters)
	add_subdirectory(47_DerivMapTest EXCLUDE_FROM_ALL)
	add_subdirectory(54_Transformations EXCLUDE_FROM_ALL)
	add_subdirectory(55_RGB18E7S3 EXCLUDE_FROM_ALL)
	add_subdirectory(61_UI)
	add_subdirectory(62_CAD EXCLUDE_FROM_ALL) # TODO: Erfan, Przemek, Francisco and co. need to resurrect this
	add_subdirectory(62_SchusslerTest EXCLUDE_FROM_ALL)
	add_subdirectory(64_EmulatedFloatTest)
	add_subdirectory(0_ImportanceSamplingEnvMaps EXCLUDE_FROM_ALL) #TODO: integrate back into 42

	add_subdirectory(66_HLSLBxDFTests EXCLUDE_FROM_ALL)
	add_subdirectory(67_RayQueryGeometry)
	add_subdirectory(68_JpegLoading)

  	add_subdirectory(70_FLIPFluids)
	add_subdirectory(71_RayTracingPipeline)

	# add new examples *before* NBL_GET_ALL_TARGETS invocation, it gathers recursively all targets created so far in this subdirectory
	NBL_GET_ALL_TARGETS(TARGETS)

	# we want to loop only over the examples so we exclude examples' interface libraries created in common subdirectory
	list(REMOVE_ITEM TARGETS ${NBL_EXAMPLES_API_TARGET} ${NBL_EXAMPLES_API_LIBRARIES})

	# we link common example api library and force examples to reuse its PCH
	foreach(T IN LISTS TARGETS)
		get_target_property(TYPE ${T} TYPE)
		if(NOT ${TYPE} MATCHES INTERFACE)
			target_link_libraries(${T} PUBLIC ${NBL_EXAMPLES_API_TARGET})
			target_include_directories(${T} PUBLIC $<TARGET_PROPERTY:${NBL_EXAMPLES_API_TARGET},INCLUDE_DIRECTORIES>)
			set_target_properties(${T} PROPERTIES DISABLE_PRECOMPILE_HEADERS OFF)
			target_precompile_headers(${T} REUSE_FROM "${NBL_EXAMPLES_API_TARGET}")

			if(NBL_EMBED_BUILTIN_RESOURCES)
				LINK_BUILTIN_RESOURCES_TO_TARGET(${T} NblExtExamplesAPIBuiltinsSource)
				LINK_BUILTIN_RESOURCES_TO_TARGET(${T} NblExtExamplesAPIBuiltinsInclude)
				LINK_BUILTIN_RESOURCES_TO_TARGET(${T} NblExtExamplesAPIBuiltinsBuild)
			endif()
		endif()
    endforeach()

	NBL_ADJUST_FOLDERS(examples)
endif()