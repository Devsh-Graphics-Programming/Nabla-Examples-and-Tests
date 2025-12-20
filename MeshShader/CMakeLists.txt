include(common RESULT_VARIABLE RES)
if(NOT RES)
        message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

if(NBL_BUILD_IMGUI)
	set(NBL_EXTRA_SOURCES
		#"${CMAKE_CURRENT_SOURCE_DIR}/src/transform.cpp" #just leaving this so i can easily reference it later
		"${CMAKE_CURRENT_SOURCE_DIR}/src/SampleApp.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/src/MeshRenderer.cpp"
	)

	set(NBL_INCLUDE_SEARCH_DIRECTORIES
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
	)

	list(APPEND NBL_LIBRARIES 
		imtestengine
		imguizmo
		"${NBL_EXT_IMGUI_UI_LIB}"
	)
	if(NBL_EMBED_BUILTIN_RESOURCES)
		set(_BR_TARGET_ ${EXECUTABLE_NAME}_builtinResourceData)
		set(RESOURCE_DIR "app_resources")

		get_filename_component(_SEARCH_DIRECTORIES_ "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
		get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
		get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)

		file(GLOB_RECURSE BUILTIN_RESOURCE_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_DIR}/*")
		foreach(RES_FILE ${BUILTIN_RESOURCE_FILES})
			LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "${RES_FILE}")
		endforeach()

		ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_SEARCH_DIRECTORIES_}" "${RESOURCE_DIR}" "nbl::this_example::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")

		LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} ${_BR_TARGET_})
	endif()
	
	
	# TODO; Arek I removed `NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET` from the last parameter here, doesn't this macro have 4 arguments anyway !?
	nbl_create_executable_project("${NBL_EXTRA_SOURCES}" "" "${NBL_INCLUDE_SEARCH_DIRECTORIES}" "${NBL_LIBRARIES}")
	# TODO: Arek temporarily disabled cause I haven't figured out how to make this target yet
	# LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} nblExamplesGeometrySpirvBRD)
endif()