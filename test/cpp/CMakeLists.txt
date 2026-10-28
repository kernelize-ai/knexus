project(knexus-test)

file(GLOB test_files "*.cpp")
foreach(file ${test_files})
  get_filename_component(test_name ${file} NAME_WE)
  add_knexus_it(NAME ${test_name}
    SRCS ${file}
    LIBS knexus-api)
endforeach()

# test_topology and test_pci exercise include/knexus/topology.h and pci.h, which need
# system hwloc (PkgConfig::HWLOC, found at the root); test_pci loads its hwloc XML
# fixtures from test/fixtures.
target_link_libraries(test_topology PRIVATE PkgConfig::HWLOC)
target_link_libraries(test_pci PRIVATE PkgConfig::HWLOC)
target_compile_definitions(test_pci PRIVATE
  KNEXUS_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../fixtures")

# test_links drives the core's link builders (src/_pcie_links.h and
# src/_fabric_links.h, private headers): PCIe with the same hwloc XML fixture
# and a temp sysfs tree, fabric with a fake runtime table.
target_link_libraries(test_links PRIVATE PkgConfig::HWLOC)
target_include_directories(test_links PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_compile_definitions(test_links PRIVATE
  KNEXUS_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../fixtures")
