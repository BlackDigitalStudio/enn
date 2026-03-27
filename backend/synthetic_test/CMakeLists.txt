# Agentic GraphRAG - Synthetic Test Project CMake
cmake_minimum_required(VERSION 3.15)
project(GraphRAGTest CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Включаем директории
include_directories(${CMAKE_SOURCE_DIR}/include)

# Собираем объектные файлы
set(SOURCES
    src/base.cpp
    src/parser.cpp
    src/graph.cpp
    main.cpp
)

# Создаем исполняемый файл
add_executable(graphrag_test ${SOURCES})

# Определения
target_compile_definitions(graphrag_test PRIVATE
    GRAPHRAG_VERSION="1.0.0"
)
