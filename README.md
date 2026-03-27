# Agentic GraphRAG

**M2M Backend для автономного LLM-агента**

Система представляет собой графовую базу знаний, объединяющую исходный код (C++ и др.) и абстрактную документацию. Это не классический векторный RAG — мы строим **математический граф зависимостей**, по которому LLM-агент алгоритмически перемещается через API.

## Архитектура

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agent     │────▶│  GraphRAG    │────▶│   Neo4j     │
│   (LLM)     │◀────│   API        │◀────│   Graph     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Qdrant     │
                    │  (vector)   │
                    └─────────────┘
```

## Быстрый старт

### 1. Запуск с Docker Compose

```bash
cd agentic-graphrag
docker-compose up -d
```

Откроется:
- **Neo4j Browser**: http://localhost:7474 (neo4j/password)
- **Qdrant Dashboard**: http://localhost:6333
- **API Docs**: http://localhost:8000/docs

### 2. Ручной запуск

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Запуск без Docker

```bash
cd backend
pip install -r requirements.txt

# Запуск только API (Neo4j должен быть доступен)
uvicorn main:app --reload
```

## API Endpoints

### Индексация

```bash
# Индексация директории
POST /api/v1/ingest
{
  "directory": "./synthetic_test",
  "extensions": [".cpp", ".h", ".hpp"]
}

# Индексация одного файла
POST /api/v1/ingest/file?file_path=./synthetic_test/main.cpp
```

### Навигация

```bash
# Получение подграфа (без кода - экономия токенов)
GET /api/v1/subgraph/{node_id}?depth=2

# Получение полного кода (только когда нужно редактировать)
GET /api/v1/node/{node_id}/code

# Поиск узлов
POST /api/v1/search
{
  "query": "parser",
  "node_type": "class"
}
```

### Метаданные

```bash
GET /api/v1/stats        # Статистика графа
GET /api/v1/types        # Допустимые типы узлов
GET /api/v1/edge-types   # Типы ребер
GET /api/v1/tags          # Допустимые теги
```

## Структура проекта

```
agentic-graphrag/
├── docker-compose.yml      # Neo4j + Qdrant
├── backend/
│   ├── main.py             # FastAPI app
│   ├── config.py           # Конфигурация
│   ├── api/
│   │   └── routes.py       # API endpoints
│   ├── graph/
│   │   ├── models.py        # GraphNode, GraphEdge
│   │   └── storage.py       # Neo4j storage
│   ├── parser/
│   │   └── cpp_parser.py    # Tree-sitter C++ parser
│   └── synthetic_test/      # Тестовый C++ проект
│       ├── include/
│       └── src/
└── README.md
```

## Фазы реализации

### Фаза A ✅ (Текущая)
- [x] AST-парсинг C++ (Tree-sitter)
- [x] Graph storage (Neo4j)
- [x] Basic API (get_subgraph, get_node_code)
- [x] Synthetic test project

### Фаза B (Следующая)
- [ ] Qdrant интеграция (векторный поиск)
- [ ] LLM-парсер для документации
- [ ] Semantic tagging

### Фаза C
- [ ] Shadow Graph (virtual_patches)
- [ ] Incremental updates
- [ ] Cascade invalidation

## Технические решения

| Компонент | Решение | Обоснование |
|-----------|---------|------------|
| Графовая БД | Neo4j | Оптимизирован для графов, Cypher запросы |
| Векторная БД | Qdrant | Легковесный, быстрый, REST API |
| AST Parser | Tree-sitter | Детерминированный парсинг без LLM |
| API Framework | FastAPI | Async, автодокументация |

## Пример использования агентом

```python
# 1. Поиск стартовой точки
nodes = api.search("parser")
start_id = nodes[0].node_id

# 2. Навигация по графу (без загрузки кода)
subgraph = api.get_subgraph(start_id, depth=2)
# Агент видит только: типы, имена, сигнатуры

# 3. Когда нужно редактировать
code = api.get_node_code(target_id)
# Загружаем полный код только выбранного узла
```

## License

MIT
