include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project(
	""
	""
	"${DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS}"
	""
	"${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}"
)