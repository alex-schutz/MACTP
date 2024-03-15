include(GoogleTest)

add_executable(
  path_tests
  path_tests.cpp
)
target_link_libraries(
  path_tests
  GTest::gtest_main
  mactp_library
)
gtest_discover_tests(path_tests)

add_executable(
  bounds_tests
  bounds_tests.cpp
)
target_link_libraries(
  bounds_tests
  GTest::gtest_main
  mactp_library
)
gtest_discover_tests(bounds_tests)
