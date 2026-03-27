/**
 * @file graph.h
 * @brief Graph Storage - хранилище графа зависимостей
 * 
 * Компонент: Использует BaseComponent
 * INCLUDES: Parser (через указатель)
 */

#pragma once

#include "base.h"
#include "parser.h"
#include <unordered_set>

namespace graphrag {

/**
 * @brief Типы ребер в графе
 */
enum class EdgeType {
    CALLS,           // Функция вызывает другую
    INHERITS,        // Класс наследует другой
    IMPLEMENTS,      // Класс реализует интерфейс
    INCLUDES,        // Файл включает другой
    DEPENDS_ON       // Общая зависимость
};

/**
 * @brief Ребро графа
 */
struct GraphEdge {
    std::string source_id;
    std::string target_id;
    EdgeType type;
    std::string metadata;
};

/**
 * @brief Узел графа
 */
struct GraphNode {
    std::string node_id;
    std::string type;          // "file", "class", "function", "variable"
    std::string name;
    std::string signature;
    std::string summary;       // Генерируется LLM (в Фазе B)
    std::string source_code;
    std::vector<std::string> tags;
    int line_start;
    int line_end;
};

/**
 * @brief Graph Storage - хранилище графа
 * 
 * IMPLEMENTS: BaseComponent
 * CALLS: ASTParser для построения графа
 * INCLUDES: напрямую использует parser.h
 */
class GraphStorage : public BaseComponent {
public:
    explicit GraphStorage(const std::string& name);
    
    // BaseComponent interface
    bool initialize() override;
    bool shutdown() override;
    std::string getVersion() const override;
    void process() override;
    
    // Операции с графом
    void addNode(const GraphNode& node);
    void addEdge(const GraphEdge& edge);
    std::vector<GraphNode> getNeighbors(const std::string& node_id, int depth = 1);
    std::vector<GraphNode> search(const std::string& query);
    
    // Пакетные операции
    void bulkAddNodes(const std::vector<GraphNode>& nodes);
    void bulkAddEdges(const std::vector<GraphEdge>& edges);
    
    // Получение графа
    std::vector<GraphNode> getAllNodes() const;
    std::vector<GraphEdge> getAllEdges() const;
    GraphNode getNode(const std::string& node_id) const;
    
    // Статистика
    size_t nodeCount() const { return nodes_.size(); }
    size_t edgeCount() const { return edges_.size(); }
    
private:
    std::string generateNodeId(const std::string& type, const std::string& name);
    void buildEdgesFromAST(const ParseResult& parse_result);
    
    std::vector<GraphNode> nodes_;
    std::vector<GraphEdge> edges_;
    std::unique_ptr<ASTParser> parser_;
    std::unordered_set<std::string> processed_files_;
};

}  // namespace graphrag
