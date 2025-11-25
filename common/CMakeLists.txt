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
        common PCH to boost its build speed to not preprocess entire Nabla again (**)
    ]]
    set_target_properties(${TARGET} PROPERTIES DISABLE_PRECOMPILE_HEADERS OFF)
    target_precompile_headers(${TARGET} REUSE_FROM "${LIB_NAME}")

    target_include_directories(${TARGET} PUBLIC "${COMMON_INCLUDE_DIRECTORY}")
    target_link_libraries(${TARGET} INTERFACE ${LIB_NAME})
endfunction()

function(REGISTER_COMMON_BUILTINS)
	cmake_parse_arguments(EX "" "TARGET;BIND;NAMESPACE" "GLOB_RGX" ${ARGN})
    get_filename_component(MOUNT_POINT "${CMAKE_CURRENT_SOURCE_DIR}/${EX_BIND}" ABSOLUTE)
    list(TRANSFORM EX_GLOB_RGX PREPEND "${MOUNT_POINT}/")
	file(GLOB_RECURSE KEYS RELATIVE "${MOUNT_POINT}" CONFIGURE_DEPENDS ${EX_GLOB_RGX})
	
    NBL_CREATE_RESOURCE_ARCHIVE(
        TARGET ${EX_TARGET}
        BIND "${MOUNT_POINT}"
        BUILTINS ${KEYS}
        NAMESPACE ${EX_NAMESPACE}
    )
    INTERFACE_TO_BUILTINS(${EX_TARGET})
endfunction()

#! common example API builtins as static library targets linked to each example
if(NBL_EMBED_BUILTIN_RESOURCES)
    REGISTER_COMMON_BUILTINS(
        TARGET NblExtExamplesAPIBuiltinsSource 
        BIND src/nbl/examples 
        NAMESPACE nbl::builtin::examples::src 
        GLOB_RGX *.hlsl *.txt
    )

    REGISTER_COMMON_BUILTINS(
        TARGET NblExtExamplesAPIBuiltinsInclude 
        BIND include/nbl/examples 
        NAMESPACE nbl::builtin::examples::include
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
target_link_libraries(${LIB_NAME} PUBLIC NblExtExamplesAPISPIRV)
if(NBL_EMBED_BUILTIN_RESOURCES)
	INTERFACE_TO_BUILTINS(NblExtExamplesAPIBuiltinsBuild)

    #[[
        we have SPIRV keys include file in examples' PCH which then gets REUSE(d) by common archives (**) in built-in mode,
        to not glitch compiler we need to ensure we inherit interface properties (include directories needed) for all targets 
        which share PCH, also note it doest really link any library, the target we inherit properties from is INTERFACE
    ]]
    target_link_libraries(NblExtExamplesAPIBuiltinsSource PUBLIC NblExtExamplesAPISPIRV)
    target_link_libraries(NblExtExamplesAPIBuiltinsInclude PUBLIC NblExtExamplesAPISPIRV)
    target_link_libraries(NblExtExamplesAPIBuiltinsBuild PUBLIC NblExtExamplesAPISPIRV)
endif()

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