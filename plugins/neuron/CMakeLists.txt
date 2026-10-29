set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/runtime_libs")

# The Neuron runtime (libnrt) and its headers, from the aws-neuronx-runtime-lib
# package. NEURON_PATH overrides the install prefix.
if(DEFINED ENV{NEURON_PATH})
    set(NEURON_PATH $ENV{NEURON_PATH} CACHE PATH "Path to which the AWS Neuron SDK has been installed")
else()
    set(NEURON_PATH "/opt/aws/neuron" CACHE PATH "Path to which the AWS Neuron SDK has been installed")
endif()

find_path(NEURON_INCLUDE_DIR nrt/nrt.h HINTS "${NEURON_PATH}/include" NO_DEFAULT_PATH)
find_library(NEURON_NRT_LIBRARY nrt HINTS "${NEURON_PATH}/lib" NO_DEFAULT_PATH)

if(NEURON_INCLUDE_DIR AND NEURON_NRT_LIBRARY)

    message(STATUS "Neuron runtime: ${NEURON_NRT_LIBRARY}")

    add_library(neuron_plugin SHARED
      neuron_buffer.cpp
      neuron_command.cpp
      neuron_library.cpp
      neuron_runtime.cpp
      neuron_schedule.cpp)

    target_include_directories(neuron_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${NEURON_INCLUDE_DIR})
    # include/knexus/pci.h (the sysfs PCI reader) needs no hwloc.
    target_link_libraries(neuron_plugin PRIVATE ${NEURON_NRT_LIBRARY} nxs-api)

    get_filename_component(NEURON_LIB_DIR ${NEURON_NRT_LIBRARY} DIRECTORY)
    set_target_properties(neuron_plugin PROPERTIES BUILD_RPATH "${NEURON_LIB_DIR}" INSTALL_RPATH "${NEURON_LIB_DIR}")

    if(UNIX AND (NOT APPLE))
        target_link_options(neuron_plugin PRIVATE "LINKER:--exclude-libs,ALL")
    endif()

else()

    if(KNEXUS_REQUIRE_NEURON)
        message(FATAL_ERROR "Neuron runtime not found under '${NEURON_PATH}' but KNEXUS_REQUIRE_NEURON=ON")
    endif()
    message(STATUS "Neuron runtime not found under '${NEURON_PATH}': neuron plugin skipped")

endif()
