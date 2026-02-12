set(SRCs
	main.cpp
	MeshLoadersApp.hpp
	MeshLoadersAppLifecycle.cpp
	MeshLoadersAppLoad.cpp
	MeshLoadersAppRuntime.cpp
	inputs.json
	README.md
)

option(NBL_MESHLOADERS_ENABLE_BENCHMARK_DATASETS "Enable benchmark dataset clone + benchmark payload setup for 12_MeshLoaders." OFF)
set(NBL_MESHLOADERS_BENCHMARK_DATASET_DIR "${CMAKE_BINARY_DIR}/meshloaders_benchmark_datasets" CACHE PATH "Destination directory for cloned 12_MeshLoaders benchmark datasets.")
set(NBL_MESHLOADERS_BENCHMARK_DATASET_REPO "https://github.com/Devsh-Graphics-Programming/Nabla-Benchmark-Datasets.git" CACHE STRING "Git repository URL for 12_MeshLoaders benchmark datasets.")
set(NBL_MESHLOADERS_BENCHMARK_PAYLOAD_RELATIVE_PATH "inputs_benchmark.json" CACHE STRING "Relative path to committed benchmark payload JSON inside dataset repo.")

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

if (NBL_MESHLOADERS_ENABLE_BENCHMARK_DATASETS)
	set(NBL_MESHLOADERS_BENCHMARK_INPUTS_JSON "${NBL_MESHLOADERS_BENCHMARK_DATASET_DIR}/${NBL_MESHLOADERS_BENCHMARK_PAYLOAD_RELATIVE_PATH}" CACHE FILEPATH "Committed benchmark testlist for 12_MeshLoaders." FORCE)
	if (NOT EXISTS "${NBL_MESHLOADERS_BENCHMARK_INPUTS_JSON}")
		if (EXISTS "${NBL_MESHLOADERS_BENCHMARK_DATASET_DIR}" AND NOT EXISTS "${NBL_MESHLOADERS_BENCHMARK_DATASET_DIR}/.git")
			message(STATUS "[meshloaders-bench] Dataset dir exists without .git, skipping fetch and using local files: ${NBL_MESHLOADERS_BENCHMARK_DATASET_DIR}")
		else()
			include(FetchContent)
			FetchContent_Declare(nbl_meshloaders_benchmark_dataset
				GIT_REPOSITORY "${NBL_MESHLOADERS_BENCHMARK_DATASET_REPO}"
				GIT_TAG "master"
				GIT_SHALLOW TRUE
				GIT_PROGRESS TRUE
				SOURCE_DIR "${NBL_MESHLOADERS_BENCHMARK_DATASET_DIR}"
				BINARY_DIR "${CMAKE_BINARY_DIR}/CMakeFiles/nbl_meshloaders_benchmark_dataset-build"
			)
			FetchContent_GetProperties(nbl_meshloaders_benchmark_dataset)
			if (NOT nbl_meshloaders_benchmark_dataset_POPULATED)
				FetchContent_Populate(nbl_meshloaders_benchmark_dataset)
			endif ()
		endif ()
	endif ()
	if (NOT EXISTS "${NBL_MESHLOADERS_BENCHMARK_INPUTS_JSON}")
		message(FATAL_ERROR "Benchmark payload JSON not found: ${NBL_MESHLOADERS_BENCHMARK_INPUTS_JSON}. Commit a full payload file into dataset repo and point NBL_MESHLOADERS_BENCHMARK_PAYLOAD_RELATIVE_PATH to it.")
	endif ()
	file(READ "${NBL_MESHLOADERS_BENCHMARK_INPUTS_JSON}" _meshloaders_payload_probe LIMIT 256)
	string(FIND "${_meshloaders_payload_probe}" "version https://git-lfs.github.com/spec/v1" _meshloaders_payload_lfs_ix)
	if (NOT _meshloaders_payload_lfs_ix EQUAL -1)
		message(FATAL_ERROR "Benchmark payload JSON must be a normal Git file, not an LFS pointer: ${NBL_MESHLOADERS_BENCHMARK_INPUTS_JSON}")
	endif ()
	message(STATUS "[meshloaders-bench] Benchmark inputs payload: ${NBL_MESHLOADERS_BENCHMARK_INPUTS_JSON}")
endif()

enable_testing()

add_test(NAME NBL_MESHLOADERS_CI
	COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" --ci
	WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
	COMMAND_EXPAND_LISTS
)

if (NBL_MESHLOADERS_ENABLE_BENCHMARK_DATASETS)
	set(_NBL_MESHLOADERS_BENCHMARK_CI_COMMON_ARGS
		--ci
		--update-references
		--testlist "${NBL_MESHLOADERS_BENCHMARK_INPUTS_JSON}"
		--loader-content-hashes
	)

	add_test(NAME NBL_MESHLOADERS_CI_BENCHMARK_HEURISTIC
		COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" ${_NBL_MESHLOADERS_BENCHMARK_CI_COMMON_ARGS} --runtime-tuning heuristic
		WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
		COMMAND_EXPAND_LISTS
	)
	add_test(NAME NBL_MESHLOADERS_CI_BENCHMARK_HYBRID
		COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" ${_NBL_MESHLOADERS_BENCHMARK_CI_COMMON_ARGS} --runtime-tuning hybrid
		WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
		COMMAND_EXPAND_LISTS
	)
	set_tests_properties(
		NBL_MESHLOADERS_CI_BENCHMARK_HEURISTIC
		NBL_MESHLOADERS_CI_BENCHMARK_HYBRID
		PROPERTIES
			TIMEOUT 21600
			LABELS "meshloaders;benchmark;ci"
	)
endif()
