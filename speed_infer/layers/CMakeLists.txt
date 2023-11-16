file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_library(atb_speed_layers SHARED ${SOURCE_FILES})
target_link_libraries(atb_speed_layers PRIVATE atb)
install(TARGETS atb_speed_layers DESTINATION lib)