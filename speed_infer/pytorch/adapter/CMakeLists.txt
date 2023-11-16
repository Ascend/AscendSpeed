file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(atb_speed_torch SHARED ${SOURCE_FILES})
target_link_libraries(atb_speed_torch PRIVATE
    atb atb_speed_core atb_speed_models atb_speed_layers torch c10
    torch_cpu torch_python torch_npu)
install(TARGETS atb_speed_torch DESTINATION lib)