set(SRCs
	main.cpp
	MeshLoadersApp.hpp
	MeshLoadersAppLifecycle.cpp
	MeshLoadersAppLoad.cpp
	MeshLoadersAppRuntime.cpp
	inputs.json
	README.md
)

set(NBL_INCLUDE_SEARCH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	"${CMAKE_SOURCE_DIR}/3rdparty"
)
set(NBL_LIBRARIES
	nlohmann_json::nlohmann_json
)

if (NBL_BUILD_MITSUBA_LOADER)
	list(APPEND NBL_INCLUDE_SEARCH_DIRECTORIES
		"${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}"
	)
	list(APPEND NBL_LIBRARIES
		"${NBL_EXT_MITSUBA_LOADER_LIB}"
	)
endif()

nbl_create_executable_project("${SRCs}" "" "${NBL_INCLUDE_SEARCH_DIRECTORIES}" "${NBL_LIBRARIES}")

if (NBL_BUILD_DEBUG_DRAW)
	target_link_libraries(${EXECUTABLE_NAME} PRIVATE Nabla::ext::DebugDraw)
endif()


add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)

add_dependencies(${EXECUTABLE_NAME} nlohmann_json::nlohmann_json)
target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INCLUDE_DIRECTORIES>)

enable_testing()

add_test(NAME NBL_MESHLOADERS_CI
	COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" --ci
	WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
	COMMAND_EXPAND_LISTS
)
