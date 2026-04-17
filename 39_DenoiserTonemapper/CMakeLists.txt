include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

option(NBL_DENOISER_TONEMAPPER_EXE_STUB_MODE "Build 39_DenoiserTonemapper as a stub executable." ON)

set(DENOISER_TONEMAPPER_EXAMPLE_LIBS
	argparse
)

set(DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS)

set(EXTRA_SOURCES
	AppInputParser.cpp
	AppInputParser.hpp
)

if(NOT NBL_DENOISER_TONEMAPPER_EXE_STUB_MODE)
	if(NOT NBL_BUILD_MITSUBA_LOADER)
		message(FATAL_ERROR "39_DenoiserTonemapper requires NBL_BUILD_MITSUBA_LOADER=ON when NBL_DENOISER_TONEMAPPER_EXE_STUB_MODE=OFF.")
	endif()

	list(APPEND DENOISER_TONEMAPPER_EXAMPLE_LIBS
		${NBL_EXT_MITSUBA_LOADER_LIB}
		${NBL_EXT_OPTIX_LIB}
		${OPTIX_DEPENDENT_LIBS}
		${MITSUBA_LOADER_DEPENDENT_LIBS}
	)
	list(APPEND DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS
		${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}
		${NBL_EXT_OPTIX_INCLUDE_DIRS}
	)
	list(APPEND EXTRA_SOURCES
		CommandLineHandler.cpp
		../../src/nbl/ext/LumaMeter/CLumaMeter.cpp
		../../src/nbl/ext/ToneMapper/CToneMapper.cpp
		../../src/nbl/ext/FFT/FFT.cpp
	)
else()
	list(APPEND EXTRA_SOURCES
		AppStub.cpp
		AppStub.hpp
	)
endif()

nbl_create_executable_project(
	"${EXTRA_SOURCES}"
	""
	"${DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS}"
	"${DENOISER_TONEMAPPER_EXAMPLE_LIBS}"
)

if(NBL_DENOISER_TONEMAPPER_EXE_STUB_MODE)
	target_compile_definitions(${EXECUTABLE_NAME} PRIVATE NBL_DENOISER_TONEMAPPER_EXE_STUB_MODE=1)
endif()

set_target_properties(${EXECUTABLE_NAME} PROPERTIES OUTPUT_NAME denoisertonemapper)
add_executable(Nabla::examples::denoiserTonemapper ALIAS ${EXECUTABLE_NAME})
