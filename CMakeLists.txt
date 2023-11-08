# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

if(NBL_BUILD_ANDROID)
	nbl_android_create_media_storage_apk()
endif()

# PCH & CommonAPI library for Nabla framework examples, must be included before example targets
add_subdirectory(common EXCLUDE_FROM_ALL)
#message(FATAL_ERROR ${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET})


# showcase the use of `nbl::core`,`nbl::system` and `nbl::asset`
add_subdirectory(01_HelloCoreSystemAsset EXCLUDE_FROM_ALL)
# showcase the use of `system::IApplicationFramework` and `nbl::video`
add_subdirectory(02_HelloCompute EXCLUDE_FROM_ALL)

add_subdirectory(03_HelloUI EXCLUDE_FROM_ALL)
# swapchain

# Nabla framework examples
add_subdirectory(01_HelloWorld EXCLUDE_FROM_ALL)
add_subdirectory(02_ComputeShader EXCLUDE_FROM_ALL)

# semi-working examples
add_subdirectory(06_MeshLoaders EXCLUDE_FROM_ALL)
add_subdirectory(07_SubpassBaking EXCLUDE_FROM_ALL)
#add_subdirectory(08.REUSE EXCLUDE_FROM_ALL)
add_subdirectory(09_ColorSpaceTest EXCLUDE_FROM_ALL)
add_subdirectory(10_AllocatorTest EXCLUDE_FROM_ALL)

#if (NBL_BUILD_BULLET AND NOT NBL_BUILD_ANDROID) # The reason why bullet shouldn't build on android: https://github.com/bulletphysics/bullet3/issues/4025
#	add_subdirectory(17_SimpleBulletIntegration EXCLUDE_FROM_ALL) # Needs example 08 to come back first
#endif()
#if (NBL_BUILD_MITSUBA_LOADER)
#	add_subdirectory(18_MitsubaLoader EXCLUDE_FROM_ALL)
#endif()
if (NBL_BUILD_MITSUBA_LOADER)
	add_subdirectory(22_RaytracedAO EXCLUDE_FROM_ALL)
endif()
# add_subdirectory(23_Autoexposure EXCLUDE_FROM_ALL)
# add_subdirectory(25_Blur EXCLUDE_FROM_ALL)
add_subdirectory(27_PLYSTLDemo EXCLUDE_FROM_ALL)
add_subdirectory(29_SpecializationConstants EXCLUDE_FROM_ALL)
# add_subdirectory(32.REUSE EXCLUDE_FROM_ALL) # TODO: @Anastazluk make this into example 04
add_subdirectory(33_Draw3DLine EXCLUDE_FROM_ALL)
add_subdirectory(34_LRUCacheUnitTest EXCLUDE_FROM_ALL)
add_subdirectory(35_GeometryCreator EXCLUDE_FROM_ALL)
# add_subdirectory(36_CUDAInterop EXCLUDE_FROM_ALL)
add_subdirectory(38_EXRSplit EXCLUDE_FROM_ALL)
# if (NBL_BUILD_MITSUBA_LOADER AND NBL_BUILD_OPTIX)
#	add_subdirectory(39_DenoiserTonemapper EXCLUDE_FROM_ALL)
# endif()
# add_subdirectory(41_VisibilityBuffer EXCLUDE_FROM_ALL) #TODO: waits for SPIRV-Cross issue about barycentrics
add_subdirectory(42_FragmentShaderPathTracer EXCLUDE_FROM_ALL)
add_subdirectory(43_SumAndCDFFilters EXCLUDE_FROM_ALL)
add_subdirectory(45_BRDFEvalTest EXCLUDE_FROM_ALL)
add_subdirectory(46_SamplingValidation EXCLUDE_FROM_ALL)
add_subdirectory(47_DerivMapTest EXCLUDE_FROM_ALL)
add_subdirectory(48_ArithmeticUnitTest EXCLUDE_FROM_ALL)
add_subdirectory(53_ComputeShaders EXCLUDE_FROM_ALL)
add_subdirectory(54_Transformations EXCLUDE_FROM_ALL)
add_subdirectory(55_RGB18E7S3 EXCLUDE_FROM_ALL)
add_subdirectory(56_RayQuery EXCLUDE_FROM_ALL)
add_subdirectory(60_ClusteredRendering EXCLUDE_FROM_ALL)
add_subdirectory(61_FilterTest EXCLUDE_FROM_ALL)
add_subdirectory(61_UI EXCLUDE_FROM_ALL)
add_subdirectory(62_CAD EXCLUDE_FROM_ALL)
add_subdirectory(62_SchusslerTest EXCLUDE_FROM_ALL)
add_subdirectory(0_ImportanceSamplingEnvMaps EXCLUDE_FROM_ALL) #TODO: integrate back into 42
add_subdirectory(64_CppCompat)


unset(NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET CACHE)