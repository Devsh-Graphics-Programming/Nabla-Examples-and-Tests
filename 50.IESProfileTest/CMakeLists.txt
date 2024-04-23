
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project("" "" "" nlohmann_json::nlohmann_json)

add_dependencies(${EXECUTABLE_NAME} nlohmann_json::nlohmann_json)
target_include_directories(${EXECUTABLE_NAME} PUBLIC $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INCLUDE_DIRECTORIES>)