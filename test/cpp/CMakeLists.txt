project(knexus-test)

file(GLOB test_files "*.cpp")
foreach(file ${test_files})
  get_filename_component(test_name ${file} NAME_WE)
  add_knexus_it(NAME ${test_name}
    SRCS ${file}
    LIBS knexus-api)
endforeach()
