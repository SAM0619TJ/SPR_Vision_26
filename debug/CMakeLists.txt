set(DEBUG_SOURCES
  web_debugger.cpp
)

add_library(debug STATIC ${DEBUG_SOURCES})

target_include_directories(debug PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(debug ${OpenCV_LIBS} pthread)

# 把 static/ 目录安装到构建目录，运行时能找到 index.html
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/static
     DESTINATION ${CMAKE_BINARY_DIR}/debug)
