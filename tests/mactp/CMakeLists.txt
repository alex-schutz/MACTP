add_executable(
  path_tests
  path_tests.cpp
)

target_link_libraries(
  path_tests
  GTest::gtest_main
  mactp_library
)

include(GoogleTest)
gtest_discover_tests(path_tests)
