project(knexus-test)

file(GLOB test_files "*.cpp")
foreach(file ${test_files})
  get_filename_component(test_name ${file} NAME_WE)
  add_knexus_it(NAME ${test_name}
    SRCS ${file}
    LIBS knexus-api)
endforeach()

# test_topology and test_pci exercise plugins/include/rt_topology.h, which needs system hwloc;
# test_pci loads its hwloc XML fixtures from test/fixtures.
find_package(PkgConfig REQUIRED)
if(NOT TARGET PkgConfig::HWLOC)
  pkg_check_modules(HWLOC REQUIRED IMPORTED_TARGET GLOBAL hwloc>=2.0)
endif()
target_link_libraries(test_topology PRIVATE PkgConfig::HWLOC)
target_link_libraries(test_pci PRIVATE PkgConfig::HWLOC)
target_compile_definitions(test_pci PRIVATE
  KNEXUS_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../fixtures")
