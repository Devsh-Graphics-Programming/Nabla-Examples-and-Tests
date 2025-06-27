#! Examples API proxy library
#[[
    We create the Nabla Examples API as a static library extension, this
    allows all examples to reuse a single precompiled header (PCH)
    instead of generating their own

    The PCH includes Nabla.h + example common interface headers and takes 
    around 1 GB per configuration, so sharing it avoids significant disk space waste
]]

nbl_create_ext_library_project(ExamplesAPI "" "${CMAKE_CURRENT_SOURCE_DIR}/src/nbl/examples/pch.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/include" "" "")

set_target_properties(${LIB_NAME} PROPERTIES DISABLE_PRECOMPILE_HEADERS OFF)
target_precompile_headers(${LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/nbl/examples/PCH.hpp")

set(COMMON_INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")

function(INTERFACE_TO_BUILTINS TARGET)
    #[[
        even though builtin target is static library its still valid to reuse 
        common PCH to boost its build speed to not preprocess entire Nabla again
    ]]
    set_target_properties(${TARGET} PROPERTIES DISABLE_PRECOMPILE_HEADERS OFF)
    target_precompile_headers(${TARGET} REUSE_FROM "${LIB_NAME}")

    target_include_directories(${TARGET} PUBLIC "${COMMON_INCLUDE_DIRECTORY}")
    target_link_libraries(${TARGET} INTERFACE ${LIB_NAME})
endfunction()

function(REGISTER_COMMON_BUILTINS)
	cmake_parse_arguments(EX "" "TARGET;ARCHIVE_ABS_ENTRY;ARCHIVE_NAMESPACE" "GLOB_RGX" ${ARGN})

    get_filename_component(INPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
    get_filename_component(OUTPUT_SRC "${CMAKE_CURRENT_BINARY_DIR}/builtin/${EX_TARGET}/src" ABSOLUTE)
    get_filename_component(OUTPUT_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/builtin/${EX_TARGET}/include" ABSOLUTE)

    set(KEYS_ENTRY "${INPUT_DIRECTORY}/${EX_ARCHIVE_ABS_ENTRY}")
    list(TRANSFORM EX_GLOB_RGX PREPEND "${KEYS_ENTRY}/")
	file(GLOB_RECURSE KEYS RELATIVE "${KEYS_ENTRY}" CONFIGURE_DEPENDS ${EX_GLOB_RGX})

    #[[
        note we do force you to specify full globbing expressions relative to keys entry which we do not filter
        because if runtime outputs .spv compilation artifacts/shader cache preprocessed.hlsl(s) to source you will hit CMake
        reconfiguration each time the file content or timestampts change and it could lead to embeding intermediate trash
    ]]
	
    unset(EXAMPLES_RESOURCES_TO_EMBED)
	foreach(KEY IN LISTS KEYS)
		LIST_BUILTIN_RESOURCE(EXAMPLES_RESOURCES_TO_EMBED "${KEY}")
	endforeach()

    ADD_CUSTOM_BUILTIN_RESOURCES(${EX_TARGET} EXAMPLES_RESOURCES_TO_EMBED "${INPUT_DIRECTORY}" "${EX_ARCHIVE_ABS_ENTRY}" "${EX_ARCHIVE_NAMESPACE}" "${OUTPUT_INCLUDE}" "${OUTPUT_SRC}")
    INTERFACE_TO_BUILTINS(${EX_TARGET})
endfunction()

#! common example API builtins as static library targets linked to each example
if(NBL_EMBED_BUILTIN_RESOURCES)
    REGISTER_COMMON_BUILTINS(TARGET NblExtExamplesAPIBuiltinsSource 
        ARCHIVE_ABS_ENTRY src/nbl/examples 
        ARCHIVE_NAMESPACE nbl::builtin::examples::src 
        GLOB_RGX *.hlsl *.txt
    )

    REGISTER_COMMON_BUILTINS(TARGET NblExtExamplesAPIBuiltinsInclude 
        ARCHIVE_ABS_ENTRY include/nbl/examples 
        ARCHIVE_NAMESPACE nbl::builtin::examples::include
        GLOB_RGX *.hpp *.h *.hlsl *.txt
    )
endif()

#! Examples API common libraries
#[[
    The rule is to avoid creating additional libraries as part of the examples' common
    interface in order to prevent generating another precompiled header (PCH) and wasting disk space

    If you have new utilities that could be shared across examples then try to implement them as header only
    and include in the PCH or in `examples.h` *if you cannot* (open the header to see details)

    but If you have a good reason to create library because you cannot make it header only 
    AND you *can REUSE* the examples' PCH then go ahead anyway and put it under `src/nbl/examples`, 
    otherwise keep it header only - a good example would be to use our embedded-whatever-you-want tool
    which does create library but can reuse example's PCH (see NblExtExamplesAPIBuiltinsSource 
    and NblExtExamplesAPIBuiltinsInclude targets)
]]

add_subdirectory("src/nbl/examples" EXCLUDE_FROM_ALL)

NBL_GET_ALL_TARGETS(TARGETS)
list(REMOVE_ITEM TARGETS ${LIB_NAME})

# the Examples API proxy library CMake target name
#[[
    this one gets linked to each executable automatically with its interface libraries
]]
set(NBL_EXAMPLES_API_TARGET ${LIB_NAME} PARENT_SCOPE)

#! names of CMake targets created in src/nbl/examples
#[[
    if your example wants to use anything from src/nbl/examples
    then you must target_link_libraries() the lib you want as we 
    don't link all those libraries to each executable automatically
]]
set(NBL_EXAMPLES_API_LIBRARIES ${TARGETS} PARENT_SCOPE)

NBL_ADJUST_FOLDERS(common)