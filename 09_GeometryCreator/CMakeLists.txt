set(NBL_INCLUDE_SERACH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)

	# TODO; Arek I removed `NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET` from the last parameter here, doesn't this macro have 4 arguments anyway !?
nbl_create_executable_project("" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "" "")
# TODO: Arek temporarily disabled cause I haven't figured out how to make this target yet
# LINK_BUILTIN_RESOURCES_TO_TARGET(${EXECUTABLE_NAME} nblExamplesGeometrySpirvBRD)