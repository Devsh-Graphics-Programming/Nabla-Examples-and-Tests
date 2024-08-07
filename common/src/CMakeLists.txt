# we add common libraries
# add_subdirectory(camera EXCLUDE_FROM_ALL)

# we get all available targets inclusive & below this directory
NBL_GET_ALL_TARGETS(NBL_SUBDIRECTORY_TARGETS)

# then we expose common include search directories to all common libraries + create link interface
foreach(NBL_TARGET IN LISTS NBL_SUBDIRECTORY_TARGETS)
    target_include_directories(${NBL_TARGET} PUBLIC $<TARGET_PROPERTY:nblCommonAPI,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(nblCommonAPI INTERFACE ${NBL_TARGET})
endforeach()

set(NBL_COMMON_API_TARGETS ${NBL_SUBDIRECTORY_TARGETS} PARENT_SCOPE)