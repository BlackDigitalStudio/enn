/**
 * @file main.cpp
 * @brief Точка входа для тестирования Graph Storage
 * 
 * Демонстрирует архитектуру:
 * - INCLUDES: base.h, parser.h, graph.h
 * - CALLS: GraphStorage methods
 * - IMPLEMENTS: BaseComponent (через наследование)
 */

#include <iostream>
#include "base.h"
#include "parser.h"
#include "graph.h"

using namespace graphrag;

int main() {
    Logger::info("=== Agentic GraphRAG Test ===");
    
    // 1. Создаем Graph Storage
    GraphStorage storage("TestGraph");
    storage.initialize();
    
    // 2. Добавляем тестовые узлы
    GraphNode file_node;
    file_node.node_id = "file:main.cpp";
    file_node.type = "file";
    file_node.name = "main.cpp";
    file_node.signature = "main.cpp";
    file_node.line_start = 1;
    file_node.line_end = 50;
    file_node.source_code = "#include <iostream>...";
    file_node.tags = {"entry_point", "test"};
    storage.addNode(file_node);
    
    // 3. Добавляем класс
    GraphNode class_node;
    class_node.node_id = "class:TestComponent";
    class_node.type = "class";
    class_node.name = "TestComponent";
    class_node.signature = "class TestComponent : public BaseComponent";
    class_node.tags = {"component", "test"};
    storage.addNode(class_node);
    
    // 4. Добавляем функцию
    GraphNode func_node;
    func_node.node_id = "function:processData";
    func_node.type = "function";
    func_node.name = "processData";
    func_node.signature = "void processData(const std::string& input)";
    func_node.tags = {"core", "processing"};
    storage.addNode(func_node);
    
    // 5. Создаем связи
    GraphEdge edge1;
    edge1.source_id = "file:main.cpp";
    edge1.target_id = "class:TestComponent";
    edge1.type = EdgeType::INCLUDES;
    storage.addEdge(edge1);
    
    GraphEdge edge2;
    edge2.source_id = "class:TestComponent";
    edge2.target_id = "function:processData";
    edge2.type = EdgeType::CALLS;
    storage.addEdge(edge2);
    
    // 6. Получаем соседей
    Logger::info("=== Getting neighbors of 'file:main.cpp' ===");
    auto neighbors = storage.getNeighbors("file:main.cpp", 2);
    Logger::info("Found " + std::to_string(neighbors.size()) + " neighbors");
    
    // 7. Поиск
    Logger::info("=== Searching for 'Test' ===");
    auto results = storage.search("Test");
    Logger::info("Found " + std::to_string(results.size()) + " matches");
    
    // 8. Статистика
    Logger::info("=== Graph Statistics ===");
    Logger::info("Nodes: " + std::to_string(storage.nodeCount()));
    Logger::info("Edges: " + std::to_string(storage.edgeCount()));
    
    storage.shutdown();
    
    Logger::info("=== Test Complete ===");
    return 0;
}
