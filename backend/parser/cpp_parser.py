"""
Agentic GraphRAG - Tree-sitter C++ Parser
Модуль для парсинга C++ кода и извлечения AST-структуры
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Set
import hashlib

from ..graph.models import GraphNode, GraphEdge, NodeType, EdgeType


@dataclass
class ParsedFile:
    """Результат парсинга файла"""
    file_path: str
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    success: bool = True
    error: str = ""


class CPPSymbolExtractor:
    """
    C++ Symbol Extractor с использованием Tree-sitter
    
    Извлекает:
    - Классы (class_specifier)
    - Функции (function_definition, method_declaration)
    - Переменные (var_declarator, parameter_declaration)
    - Наследования (base_class_clause)
    - Вызовы функций (call_expression)
    - Include директивы
    """
    
    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self._try_import_tree_sitter()
    
    def _try_import_tree_sitter(self):
        """Попытка импортировать tree-sitter + tree-sitter-cpp"""
        try:
            import tree_sitter as ts
            import tree_sitter_cpp as tscpp
            self.parser_available = True
            self._ts_language = ts.Language(tscpp.language())
            self._ts_parser = ts.Parser(self._ts_language)
        except (ImportError, Exception):
            self.parser_available = False
    
    def _generate_node_id(self, node_type: str, name: str, file_path: str) -> str:
        """Генерация уникального ID узла"""
        content = f"{node_type}:{name}:{file_path}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{node_type}::{name}#{hash_suffix}"
    
    def parse_file(self, file_path: str) -> ParsedFile:
        """
        Парсит C++ файл и возвращает список узлов и ребер
        """
        result = ParsedFile(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            result.success = False
            result.error = str(e)
            return result
        
        if self.parser_available:
            return self._parse_with_tree_sitter(file_path, content)
        else:
            return self._parse_with_regex(file_path, content)
    
    def _parse_with_tree_sitter(self, file_path: str, content: str) -> ParsedFile:
        """Парсинг с использованием tree-sitter"""
        content_bytes = content.encode("utf8")
        tree = self._ts_parser.parse(content_bytes)

        result = ParsedFile(file_path=file_path)

        # Обход дерева — передаем bytes для корректных offset
        self._walk_tree(tree.root_node, content_bytes, result, file_path)

        return result
    
    def _text(self, node, content_bytes: bytes) -> str:
        """Извлечение текста узла из bytes"""
        return content_bytes[node.start_byte:node.end_byte].decode("utf8", errors="replace")

    def _walk_tree(self, node, content_bytes: bytes, result: ParsedFile, file_path: str):
        """Рекурсивный обход AST"""
        node_type = node.type

        # Обработка include директив
        if node_type == "preproc_include":
            import_path = self._extract_include_path(node, content_bytes)
            if import_path:
                result.imports.append(import_path)

        # Обработка классов
        elif node_type == "class_specifier":
            class_info = self._extract_class(node, content_bytes, file_path, result)
            if class_info:
                result.nodes.append(class_info)

        # Обработка функций
        elif node_type in ("function_definition",):
            func_info = self._extract_function(node, content_bytes, file_path)
            if func_info:
                result.nodes.append(func_info)

        # Обработка структур
        elif node_type == "struct_specifier":
            struct_info = self._extract_struct(node, content_bytes, file_path)
            if struct_info:
                result.nodes.append(struct_info)

        # Рекурсия в детей
        for child in node.children:
            self._walk_tree(child, content_bytes, result, file_path)
    
    def _extract_class(self, node, content_bytes: bytes, file_path: str, result: ParsedFile) -> Optional[GraphNode]:
        """Извлечение информации о классе"""
        name_node = self._find_child(node, "type_identifier") or self._find_child(node, "identifier")
        if not name_node:
            return None

        name = self._text(name_node, content_bytes).strip()

        # Извлечение базовых классов (наследование)
        base_classes = []
        for child in node.children:
            if child.type == "base_class_clause":
                bases = self._extract_base_classes(child, content_bytes)
                base_classes.extend(bases)

        summary = f"Class {name}"
        if base_classes:
            summary += f" : {', '.join(base_classes)}"

        node_obj = GraphNode(
            node_id=self._generate_node_id("class", name, file_path),
            type=NodeType.CLASS.value,
            name=name,
            signature=f"class {name}",
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            source_code=self._text(node, content_bytes),
            summary=summary,
            tags=["core"] if not base_classes else ["concrete"]
        )

        for base in base_classes:
            edge = GraphEdge(
                source_id=node_obj.node_id,
                target_id=self._generate_node_id("class", base, file_path),
                edge_type=EdgeType.INHERITS.value
            )
            result.edges.append(edge)

        return node_obj
    
    def _extract_function(self, node, content_bytes: bytes, file_path: str) -> Optional[GraphNode]:
        """Извлечение информации о функции"""
        declarator = self._find_child(node, "function_declarator")
        if not declarator:
            return None

        name_node = (
            self._find_child(declarator, "identifier")
            or self._find_child(declarator, "field_identifier")
            or self._find_child(declarator, "qualified_identifier")
        )
        if not name_node:
            return None

        name = self._text(name_node, content_bytes).strip()

        # Сигнатура: всё кроме тела функции
        body = self._find_child(node, "compound_statement")
        if body:
            signature = content_bytes[node.start_byte:body.start_byte].decode("utf8", errors="replace").strip()
        else:
            signature = self._text(node, content_bytes).strip()

        return GraphNode(
            node_id=self._generate_node_id("function", name, file_path),
            type=NodeType.FUNCTION.value,
            name=name,
            signature=signature,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            source_code=self._text(node, content_bytes),
            summary=f"Function {name}",
            tags=["core"]
        )
    
    def _extract_struct(self, node, content_bytes: bytes, file_path: str) -> Optional[GraphNode]:
        """Извлечение информации о структуре"""
        name_node = self._find_child(node, "type_identifier")
        if not name_node:
            return None

        name = self._text(name_node, content_bytes).strip()

        return GraphNode(
            node_id=self._generate_node_id("struct", name, file_path),
            type=NodeType.STRUCT.value,
            name=name,
            signature=f"struct {name}",
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            source_code=self._text(node, content_bytes),
            summary=f"Struct {name}",
            tags=["data"]
        )
    
    def _find_child(self, node, child_type: str):
        """Поиск потомка по типу"""
        for child in node.children:
            if child.type == child_type:
                return child
        return None
    
    def _extract_include_path(self, node, content_bytes: bytes) -> Optional[str]:
        """Извлечение пути из #include"""
        for child in node.children:
            if child.type in ("string_literal", "system_lib_string"):
                raw = self._text(child, content_bytes)
                # Strip surrounding <> or ""
                return raw.strip('<>"')
        return None

    def _extract_base_classes(self, node, content_bytes: bytes) -> List[str]:
        """Извлечение базовых классов из наследования"""
        bases = []
        for child in node.children:
            if child.type == "type_identifier":
                bases.append(self._text(child, content_bytes).strip())
        return bases
    
    def _find_descendant(self, node, child_type: str):
        """Рекурсивный поиск потомка по типу"""
        for child in node.children:
            if child.type == child_type:
                return child
            found = self._find_descendant(child, child_type)
            if found:
                return found
        return None
    
    def _parse_with_regex(self, file_path: str, content: str) -> ParsedFile:
        """
        Fallback парсер на регулярных выражениях
        Используется если tree-sitter недоступен
        """
        import re
        
        result = ParsedFile(file_path=file_path)
        lines = content.split('\n')
        
        # Паттерны для C++
        class_pattern = re.compile(r'class\s+(\w+)(?:\s*:\s*([^{]+))?')
        func_pattern = re.compile(r'(?:virtual\s+)?(?:void|int|bool|auto|std::\w+|<[^>]+>)?\s*(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*(?:=|0)?\s*\{')
        struct_pattern = re.compile(r'struct\s+(\w+)')
        include_pattern = re.compile(r'#include\s*[<"]([^>"]+)[>"]')
        
        for i, line in enumerate(lines, 1):
            # Include
            match = include_pattern.search(line)
            if match:
                result.imports.append(match.group(1))
            
            # Class
            match = class_pattern.search(line)
            if match:
                name = match.group(1)
                bases = match.group(2) or ""
                bases = bases.strip()
                
                node = GraphNode(
                    node_id=self._generate_node_id("class", name, file_path),
                    type=NodeType.CLASS.value,
                    name=name,
                    signature=f"class {name}" + (f" : {bases}" if bases else ""),
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    summary=f"Class {name}" + (f" inherits {bases}" if bases else ""),
                    tags=["core"]
                )
                result.nodes.append(node)
            
            # Function (упрощенный)
            match = func_pattern.search(line)
            if match:
                name = match.group(1)
                if name and name not in ("if", "while", "for", "switch"):
                    node = GraphNode(
                        node_id=self._generate_node_id("function", name, file_path),
                        type=NodeType.FUNCTION.value,
                        name=name,
                        signature=match.group(0).strip(),
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        summary=f"Function {name}",
                        tags=["core"]
                    )
                    result.nodes.append(node)
            
            # Struct
            match = struct_pattern.search(line)
            if match:
                name = match.group(1)
                node = GraphNode(
                    node_id=self._generate_node_id("struct", name, file_path),
                    type="struct",
                    name=name,
                    signature=f"struct {name}",
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    summary=f"Struct {name}",
                    tags=["data"]
                )
                result.nodes.append(node)
        
        return result


def scan_directory(root_dir: str, extensions: List[str] = None) -> List[str]:
    """Сканирование директории и поиск файлов по расширению"""
    if extensions is None:
        extensions = ['.cpp', '.h', '.hpp', '.cc']
    
    files = []
    for ext in extensions:
        for root, dirs, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(ext):
                    files.append(os.path.join(root, filename))
    return files


# Тест
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Тестовый файл по умолчанию
        test_file = os.path.join(os.path.dirname(__file__), "synthetic_test", "main.cpp")
    
    print(f"Parsing: {test_file}")
    
    parser = CPPSymbolExtractor(os.path.dirname(os.path.dirname(__file__)))
    result = parser.parse_file(test_file)
    
    print(f"\n=== Parsed: {test_file} ===")
    print(f"Success: {result.success}")
    print(f"Nodes: {len(result.nodes)}")
    print(f"Edges: {len(result.edges)}")
    print(f"Imports: {result.imports}")
    
    for node in result.nodes:
        print(f"\n  [{node.type}] {node.name}")
        print(f"    ID: {node.node_id}")
        print(f"    Signature: {node.signature}")
