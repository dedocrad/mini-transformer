# mini-transformer

Mini-transformer, который переворачивает строки, состоящие из латинских букв. Проект создан для изучения архитектуры Transformer и механизма attention.

## Запуск через Docker

Понадобится установленный и запущенный Docker Desktop.

Соберите образы и запустите приложение:

```bash
docker compose up --build
```

После запуска:

- Streamlit-приложение: http://localhost:8501
- MLflow UI: http://localhost:5001

Streamlit и MLflow используют общий именованный Docker volume `mlflow-data`, поэтому логи экспериментов сохраняются между перезапусками контейнеров.

Остановить контейнеры:

```bash
docker compose down
```

Чтобы удалить контейнеры вместе с сохраненными данными MLflow:

```bash
docker compose down -v
```

## Структура проекта

- `src/` — модель, токенизатор, inference и Streamlit-приложение.
- `models/` — сохраненные веса модели.
- `data/` — данные для обучения.
- `notebooks/` — ноутбуки для подготовки данных, обучения и inference.
- `Dockerfile` и `docker-compose.yml` — конфигурация контейнеров.

## TODO:
- структурировать `dataset.ipynb`
- дооформить веб-приложение `app.py`