if(NBL_BUILD_IMGUI)
set(SRCs
	AppInit.cpp AppRender.cpp AppGPU.cpp AppUI.cpp AppEvent.cpp AppInputParser.cpp
	App.hpp AppInputParser.hpp
	IES.cpp IES.hpp
	CSimpleIESRenderer.hpp
	inputs.json
)

set(LIBs 
	imtestengine
	imguizmo
	"${NBL_EXT_IMGUI_UI_LIB}"
)
	
nbl_create_executable_project("${SRCs}" "" "" "${LIBs}")
target_link_libraries(${EXECUTABLE_NAME} PRIVATE nlohmann_json::nlohmann_json)
add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(DEPENDS
	app_resources/common.hlsl
	app_resources/false_color.hlsl
	app_resources/imgui.opts.hlsl
	app_resources/ies.unified.hlsl
	app_resources/imgui.unified.hlsl
)
target_sources(${EXECUTABLE_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

set(SM 6_8)
set(JSON [=[
[
	{
		"INPUT": "app_resources/ies.unified.hlsl",
		"KEY": "ies.unified"
	},
	{
		"INPUT": "app_resources/imgui.unified.hlsl",
		"KEY": "imgui.unified"
	}
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
	-I "${NBL_ROOT_PATH}/include" # a workaround due to imgui ext headers which are not part of Nabla builtin archive
	-I "${CMAKE_CURRENT_SOURCE_DIR}"
	-O3
	-T lib_${SM}
)

NBL_CREATE_NSC_COMPILE_RULES(
	TARGET ${EXECUTABLE_NAME}SPIRV
	LINK_TO ${EXECUTABLE_NAME}
	DEPENDS ${DEPENDS}
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
endif()
