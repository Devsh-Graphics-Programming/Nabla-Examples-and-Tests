include(common RESULT_VARIABLE RES)
if(NOT RES)
		message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

if(NBL_BUILD_IMGUI AND NBL_BUILD_DEBUG_DRAW)
	set(NBL_INCLUDE_SERACH_DIRECTORIES
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
		"${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}"
	)

	list(APPEND NBL_LIBRARIES 
		imtestengine
		imguizmo
		"${NBL_EXT_IMGUI_UI_LIB}"
		"${NBL_EXT_MITSUBA_LOADER_LIB}"
	)

	nbl_create_executable_project("" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")

	target_link_libraries(${EXECUTABLE_NAME} PRIVATE Nabla::ext::DebugDraw)
endif()

