if(NBL_BUILD_IMGUI)
	set(NBL_EXTRA_SOURCES
		"${CMAKE_CURRENT_SOURCE_DIR}/src/transform.cpp"
	)

	set(NBL_INCLUDE_SERACH_DIRECTORIES
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
	)

	list(APPEND NBL_LIBRARIES 
		imtestengine
		imguizmo
		"${NBL_EXT_IMGUI_UI_LIB}"
	)
	
	# TODO; Arek I removed `NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET` from the last parameter here, doesn't this macro have 4 arguments anyway !?
	nbl_create_executable_project("${NBL_EXTRA_SOURCES}" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")
	# TODO: Arek temporarily disabled cause I haven't figured out how to make this target yet
	# LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} nblExamplesGeometrySpirvBRD)
endif()