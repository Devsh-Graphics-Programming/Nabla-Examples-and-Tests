if(NBL_BUILD_IMGUI)
set(SRCs
	AppInputParser.cpp AppInputParser.hpp
	inputs.json
)

set(LIBs 
	imtestengine
	imguizmo
	"${NBL_EXT_IMGUI_UI_LIB}"
)
	
nbl_create_executable_project("${SRCs}" "" "" "${LIBs}")
target_link_libraries(${EXECUTABLE_NAME} PRIVATE nlohmann_json::nlohmann_json)

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(DEPENDS
	app_resources/common.hlsl
	app_resources/compute.hlsl
	app_resources/pixel.hlsl
	app_resources/vertex.hlsl
	app_resources/imgui.vertex.hlsl
	app_resources/imgui.pixel.hlsl
	app_resources/imgui.opts.hlsl
)
target_sources(${EXECUTABLE_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

set(SM 6_8)
set(JSON [=[
[
    {
		"INPUT": "app_resources/compute.hlsl",
		"KEY": "compute",
        "COMPILE_OPTIONS": ["-T", "lib_${SM}"],
		"DEPENDS": [],
		"CAPS": []
    },
	{
		"INPUT": "app_resources/pixel.hlsl",
		"KEY": "pixel",
		"COMPILE_OPTIONS": ["-T", "lib_${SM}"],
		"DEPENDS": [],
		"CAPS": []
	},
	{
		"INPUT": "app_resources/vertex.hlsl",
		"KEY": "vertex",
		"COMPILE_OPTIONS": ["-T", "lib_${SM}"],
		"DEPENDS": [],
		"CAPS": []
	},
	{
		"INPUT": "app_resources/imgui.vertex.hlsl",
		"KEY": "imgui.vertex",
		"COMPILE_OPTIONS": ["-T", "lib_${SM}"],
		"DEPENDS": [],
		"CAPS": []
	},
	{
		"INPUT": "app_resources/imgui.pixel.hlsl",
		"KEY": "imgui.pixel",
		"COMPILE_OPTIONS": ["-T", "lib_${SM}"],
		"DEPENDS": [],
		"CAPS": []
	}
]
]=])
string(CONFIGURE "${JSON}" JSON)

NBL_CREATE_NSC_COMPILE_RULES(
	TARGET ${EXECUTABLE_NAME}SPIRV
	LINK_TO ${EXECUTABLE_NAME}
	DEPENDS ${DEPENDS}
	BINARY_DIR ${OUTPUT_DIRECTORY}
	MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
	COMMON_OPTIONS -I ${CMAKE_CURRENT_SOURCE_DIR} -O3
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