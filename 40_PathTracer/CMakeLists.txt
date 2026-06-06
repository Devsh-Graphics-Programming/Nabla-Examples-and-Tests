include(common)

set(NBL_INCLUDE_SERACH_DIRECTORIES
	"${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}"
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
	"${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"
	"${NBL_ROOT_PATH}/3rdparty/portable-file-dialogs"
	"${CMAKE_CURRENT_SOURCE_DIR}/app_resources"
)
set(NBL_LIBRARIES
	"${NBL_EXT_MITSUBA_LOADER_LIB}"
	Nabla::ext::FullScreenTriangle
    imguizmo
	"${NBL_EXT_IMGUI_UI_LIB}"
)
set(NBL_EXAMPLE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/io/CSceneLoader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/gui/CSceneWindow.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/gui/CSessionWindow.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/gui/CUIManager.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/SAASequence.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/CScene.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/CRenderer.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/CSession.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/resolve/CBasicRWMCResolver.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/resolve/IResolver.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/present/CWindowPresenter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/present/IPresenter.h"

    "${CMAKE_CURRENT_SOURCE_DIR}/src/io/CSceneLoader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CSession.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CScene.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CRenderer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/CLightTree.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/resolve/CBasicRWMCResolver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/present/CWindowPresenter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/CUIManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/CSceneWindow.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/CSessionWindow.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/common.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/io/CSceneLoader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/CLightTree.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/CRenderer.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/CScene.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/CSession.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/SAASequence.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/present/CWindowPresenter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/present/IPresenter.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/resolve/CBasicRWMCResolver.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/resolve/IResolver.h"

    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CPathTracerBenchmark.h"

    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/bda_accessors.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/common.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/light_tree.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/scene.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/session.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/pathtrace/common.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/pathtrace/push_constants.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/present/push_constants.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/renderer/shaders/resolve/rwmc.hlsl"

    "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/pathtrace/beauty.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/pathtrace/common.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/pathtrace/debug.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/pathtrace/nee_proposal_probe.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/pathtrace/next_event_estimator.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/pathtrace/previs.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/app_resources/present/default.hlsl"
)
nbl_create_executable_project("${NBL_EXAMPLE_SOURCES}" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")

target_compile_definitions(${EXECUTABLE_NAME} PRIVATE)

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")

set(SM 6_9)
set(JSON [=[
[
    {
        "INPUT": "app_resources/pathtrace/previs.hlsl",
        "KEY": "pathtrace_previs",
    },
    {
        "INPUT": "app_resources/pathtrace/beauty.hlsl",
        "KEY": "pathtrace_beauty",
    },
    {
        "INPUT": "app_resources/pathtrace/beauty.hlsl",
        "KEY": "pathtrace_beauty_nee_only",
        "COMPILE_OPTIONS": ["-DNBL_MIS_MODE=0"],
    },
    {
        "INPUT": "app_resources/pathtrace/beauty.hlsl",
        "KEY": "pathtrace_beauty_bxdf_only",
        "COMPILE_OPTIONS": ["-DNBL_MIS_MODE=1"],
    },
    {
        "INPUT": "app_resources/pathtrace/beauty.hlsl",
        "KEY": "pathtrace_beauty_tree",
        "COMPILE_OPTIONS": ["-DNBL_NEE_USE_ALIAS=0"],
    },
    {
        "INPUT": "app_resources/pathtrace/beauty.hlsl",
        "KEY": "pathtrace_beauty_nee_only_tree",
        "COMPILE_OPTIONS": ["-DNBL_MIS_MODE=0", "-DNBL_NEE_USE_ALIAS=0"],
    },
    {
        "INPUT": "app_resources/pathtrace/debug.hlsl",
        "KEY": "pathtrace_debug",
    },
	{
        "INPUT": "app_resources/present/default.hlsl",
        "KEY": "present_default",
    }
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
    -I "${CMAKE_CURRENT_SOURCE_DIR}/../common/include"
    -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
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


