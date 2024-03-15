add_library(mactp_library simulator.cpp statespace.cpp path.cpp bounds.cpp)
target_link_libraries(mactp_library mcjesp_library)
