set(NBL_INCLUDE_SERACH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)

nbl_create_executable_project("" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")
LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} geometryCreatorSpirvBRD)