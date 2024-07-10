if(NBL_BUILD_IMGUI)
	list(APPEND NBL_LIBRARIES 
		imtestengine
		imguizmo
		"${NBL_EXT_IMGUI_UI_LIB}"
	)

	nbl_create_executable_project("" "" "" "${NBL_LIBRARIES}" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")
endif()