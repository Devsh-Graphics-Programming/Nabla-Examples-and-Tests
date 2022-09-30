include(common)

set(ASSET_DIR ${PROJECT_SOURCE_DIR}/examples_tests/media) 
nbl_create_executable_project("" "" "" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}" ${ASSET_DIR})
