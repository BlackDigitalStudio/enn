/**
 * @file parser.h
 * @brief AST Parser для исходного кода
 * 
 * Компонент: Использует BaseComponent
 * Зависимости: ВЫЗЫВАЕТ BaseComponent методы
 */

#pragma once

#include "base.h"
#include <unordered_map>

namespace graphrag {

/**
 * @brief Результат парсинга AST-узла
 */
struct ASTNode {
    std::string type;           // "class", "function", "variable"
    std::string name;
    std::string signature;       // Полная сигнатура
    int line_start;
    int line_end;
    std::vector<ASTNode> children;
    std::vector<std::string> calls;  // Вызываемые функции
};

/**
 * @brief Результат парсинга файла
 */
struct ParseResult {
    std::string file_path;
    std::vector<ASTNode> nodes;
    std::vector<std::string> imports;
    bool success;
    std::string error_message;
};

/**
 * @brief AST Parser - парсит исходный код и извлекает структуру
 * 
 * IMPLEMENTS: BaseComponent
 * CALLS: Logger::debug
 */
class ASTParser : public BaseComponent {
public:
    explicit ASTParser(const std::string& name);
    
    // BaseComponent interface
    bool initialize() override;
    bool shutdown() override;
    std::string getVersion() const override;
    void process() override;
    
    // Специфичные методы
    ParseResult parseFile(const std::string& file_path);
    ParseResult parseCode(const std::string& code, const std::string& language);
    std::vector<ASTNode> extractFunctions(const ASTNode& root);
    std::vector<ASTNode> extractClasses(const ASTNode& root);
    
private:
    bool validateSyntax(const std::string& code);
    ASTNode buildAST(const std::string& code);
    
    std::unordered_map<std::string, int> node_count_;
    int max_depth_;
};

/**
 * @brief Фабрика парсеров для разных языков
 */
class ParserFactory {
public:
    static std::unique_ptr<ASTParser> createParser(const std::string& language);
};

}  // namespace graphrag
