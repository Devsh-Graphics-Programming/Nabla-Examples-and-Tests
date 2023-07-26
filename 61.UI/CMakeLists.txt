if(NBL_BUILD_IMGUI)

	include(common RESULT_VARIABLE RES)
	if(NOT RES)
		message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
	endif()

	list(APPEND libs 
		"imgui"
		"${NBL_EXT_IMGUI_UI_LIB}"
	)

nbl_create_executable_project("" "" "" "${libs}" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

endif()