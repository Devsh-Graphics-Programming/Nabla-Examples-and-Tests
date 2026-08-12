include(common)

set(NBL_INCLUDE_SERACH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

if(NOT NBL_BUILD_IMGUI)
	return()
endif()

set(NBL_INCLUDE_SEARCH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/app_resources"
)

set(NBL_LIBRARIES
    imguizmo
	"${NBL_EXT_IMGUI_UI_LIB}"
	Nabla::ext::FullScreenTriangle
)

nbl_create_executable_project(
	""                                  # extra .cpp files (none)
	""                                  # extra compile options (none)
	"${NBL_INCLUDE_SEARCH_DIRECTORIES}"
	"${NBL_LIBRARIES}"
	"${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}"
)

## ---------------------- Shaders setup ---------------------------

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(SM 6_9)

set(JSON [=[
[
    {
        "INPUT": "app_resources/present.frag.hlsl",
        "KEY": "present_frag",
    },
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
    -I "${CMAKE_CURRENT_SOURCE_DIR}/../common/include"
    -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
    -I "${CMAKE_CURRENT_SOURCE_DIR}"
    -T lib_${SM}
)

NBL_CREATE_NSC_COMPILE_RULES(
    TARGET ${EXECUTABLE_NAME}SPIRV
    LINK_TO ${EXECUTABLE_NAME}
    BINARY_DIR ${OUTPUT_DIRECTORY}
    MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
    COMMON_OPTIONS ${COMPILE_OPTIONS}
    OUTPUT_VAR KEYS
    INCLUDE nbl/this_example/builtin/build/spirv/keys.hpp
    NAMESPACE nbl::this_example::builtin::build
    INPUTS ${JSON}
)

NBL_CREATE_RESOURCE_ARCHIVE(
    NAMESPACE nbl::this_example::builtin::build
    TARGET ${EXECUTABLE_NAME}_builtinsBuild
    LINK_TO ${EXECUTABLE_NAME}
    BIND ${OUTPUT_DIRECTORY}
    BUILTINS ${KEYS}
)
