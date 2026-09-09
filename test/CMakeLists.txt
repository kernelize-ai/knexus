# Google Test
include_directories(../third_party/googletest)

enable_testing()
include(GoogleTest)

function(add_knexus_it)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS LIBS DEFS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  get_property(knexus_libs GLOBAL PROPERTY KNEXUS_LIBRARIES)

  add_test(NAME ${__NAME}
          COMMAND ${__NAME})
  add_executable(
          ${__NAME}
          ${__SRCS})
  target_link_libraries(
          ${__NAME}
          PRIVATE
          GTest::gtest_main
          ${knexus_libs}
          gmock
          ${__LIBS})

  if(NOT MSVC)
    target_compile_options(${__NAME} PRIVATE -fno-rtti)
  endif()

  target_compile_definitions(${__NAME} PRIVATE ${__DEFS})

  # Without the TEST_DISCOVERY_TIMEOUT, the tests randomly time out on my mac
  # laptop.  I think the issue may be that the very first time you run a program
  # it's a bit slow.
  gtest_discover_tests(${__NAME} DISCOVERY_TIMEOUT 60)

  # Add the unit test to the top-level unit test target.
  # add_dependencies(KNexusIntegrationTest ${__NAME})
endfunction()

# Device-library schema validation. Pure Python: needs neither a build artifact nor the
# knexus extension module, only `jsonschema` (see requirements.txt), so it runs anywhere.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME device_schema
           COMMAND ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/python/test_device_schema.py)
  # Provenance audit of the Derived figures the schema validator cannot check.
  # Pure stdlib Python: not even jsonschema.
  add_test(NAME derived_provenance
           COMMAND ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/python/test_derived_provenance.py)
else()
  message(STATUS "Python3 interpreter not found; skipping the device_schema and "
                 "derived_provenance tests")
endif()

add_subdirectory(kernels)
add_subdirectory(cpp)
#add_subdirectory(pyknexus)
