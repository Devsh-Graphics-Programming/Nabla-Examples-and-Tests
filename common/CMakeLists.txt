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

#! Examples API common libraries
#[[
    The rule is to avoid creating additional libraries as part of the examples' common
    interface in order to prevent generating another precompiled header (PCH) and wasting disk space

    If you have new utilities that could be shared across examples then try to implement them as header only
    and include in the PCH or in `examples.h` *if you cannot* (open the header to see details)

    but If you have a good reason to create library because you cannot make it header only 
    AND you *can REUSE* the examples' PCH then go ahead anyway and put it under `src/nbl/examples`, 
    otherwise keep it header only - a good example would be to use our embedded-whatever-you-want tool
    which does create library but can reuse example's PCH
]]

#! NOTE: as I write it we don't have any targets there yet
add_subdirectory("src/nbl/examples" EXCLUDE_FROM_ALL)

NBL_GET_ALL_TARGETS(TARGETS)
list(REMOVE_ITEM TARGETS ${LIB_NAME})

# the Examples API proxy library CMake target name
#[[
    this one gets linked to each executable automatically
]]
set(NBL_EXAMPLES_API_TARGET ${LIB_NAME} PARENT_SCOPE)

#! names of CMake targets created in src/nbl/examples
#[[
    if your example wants to use anything from src/nbl/examples
    then you must target_link_libraries() the lib you want as we 
    don't link all those libraries to each executable automatically
]]
set(NBL_EXAMPLES_API_LIBRARIES ${TARGETS} PARENT_SCOPE)