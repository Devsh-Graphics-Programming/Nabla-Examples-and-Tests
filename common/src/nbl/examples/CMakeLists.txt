# TODO: @AnastaZluk redo the PCH
# add_subdirectory(pch EXCLUDE_FROM_ALL)

# we add common libraries
# add_subdirectory(cameras EXCLUDE_FROM_ALL) # header only currently
add_subdirectory(geometry EXCLUDE_FROM_ALL)

# we get all available targets inclusive & below this directory
NBL_GET_ALL_TARGETS(NBL_SUBDIRECTORY_TARGETS)

# then we expose common include search directories to all common libraries + create link interface
foreach(NBL_TARGET IN LISTS NBL_SUBDIRECTORY_TARGETS)
    target_include_directories(${NBL_TARGET} PUBLIC $<TARGET_PROPERTY:nblExamplesAPI,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(nblExamplesAPI INTERFACE ${NBL_TARGET})
endforeach()

#
set(NBL_COMMON_API_TARGETS ${NBL_SUBDIRECTORY_TARGETS} PARENT_SCOPE)