set(NBL_INCLUDE_SERACH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)
set(NBL_LIBRARIES)

if (NBL_BUILD_MITSUBA_LOADER)
	list(APPEND NBL_INCLUDE_SERACH_DIRECTORIES
		"${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}"
	)
	list(APPEND NBL_LIBRARIES
		"${NBL_EXT_MITSUBA_LOADER_LIB}"
	)
endif()

	# TODO; Arek I removed `NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET` from the last parameter here, doesn't this macro have 4 arguments anyway !?
nbl_create_executable_project("" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")
# TODO: Arek temporarily disabled cause I haven't figured out how to make this target yet
# LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} nblExamplesGeometrySpirvBRD)

if (NBL_BUILD_DEBUG_DRAW)
	add_dependencies(${EXECUTABLE_NAME} ${NBL_EXT_DEBUG_DRAW_TARGET})
	target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${NBL_EXT_DEBUG_DRAW_TARGET})
	target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:${NBL_EXT_DEBUG_DRAW_TARGET},INCLUDE_DIRECTORIES>)
endif()


add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)