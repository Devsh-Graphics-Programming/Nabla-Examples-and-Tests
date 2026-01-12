set(NBL_INCLUDE_SERACH_DIRECTORIES
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)
set(NBL_LIBRARIES)

nbl_create_executable_project("" "" "${NBL_INCLUDE_SERACH_DIRECTORIES}" "${NBL_LIBRARIES}")

target_link_libraries(${EXECUTABLE_NAME} PRIVATE Nabla::ext::EnvmapImportanceSampling)
