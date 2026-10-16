# Regression test for `InputSystem::tryGetDefault*`, needs neither a GPU nor a display.
# Run with `ctest --test-dir <build>/examples_tests/common/test`.
enable_testing()

add_executable(nbl_examples_input_system_test CInputSystemTest.cpp)
target_link_libraries(nbl_examples_input_system_test PRIVATE Nabla)
target_compile_features(nbl_examples_input_system_test PRIVATE cxx_std_20)

# examples_tests/CMakeLists.txt links every target under it to the examples API and reuses its PCH,
# so build with the same flags as the examples (see `nbl_create_executable_project`)
nbl_adjust_flags(TARGET nbl_examples_input_system_test MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
nbl_adjust_definitions()
if(NOT WIN32 AND NOT ANDROID AND CMAKE_POSITION_INDEPENDENT_CODE)
	set_target_properties(nbl_examples_input_system_test PROPERTIES POSITION_INDEPENDENT_CODE OFF)
	target_compile_options(nbl_examples_input_system_test PRIVATE -fPIC)
endif()

add_test(NAME NBL_EXAMPLES_INPUT_SYSTEM COMMAND $<TARGET_FILE:nbl_examples_input_system_test>)
set_tests_properties(NBL_EXAMPLES_INPUT_SYSTEM PROPERTIES TIMEOUT 60)
