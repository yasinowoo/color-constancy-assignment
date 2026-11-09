# color-constancy-assignment

## Установка uv

Проект использует пакетный менеджер [uv](https://docs.astral.sh/uv/).

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Альтернативно: `pip install uv`. Проверить установку: `uv --version`.

Домашнее задание посвящено двум этапам конвейера обработки изображений: балансу белого (Automatic White Balance) и преобразованию цветовых координат (Color Space Transform). Помимо классических методов AWB, вы дообучите нейросетевую модель FC4.

## Данные

1. Скачайте архив с данными:   **https://www.kaggle.com/datasets/yasinowo/isp-assignment-dataset**
2. Распакуйте в корне репозитория

После распаковки структура должна выглядеть так:

```
color-constancy-assignment/
├── data/
│   ├── hsi/
│   │   ├── 0.npy            # ГСИ + маски мишени и белого патча
│   │   ├── 1.npy
│   │   └── ...
│   └── illuminants.npy      # спектры источников для аугментации (FC4)
├── tests/
├── awb.py
├── cst.py
├── fc4.py
├── metrics.py
├── preprocess.py
├── lab.ipynb
└── pyproject.toml
```

## Что нужно сделать

- **Реализовать** заготовки в `awb.py`, `cst.py`, `metrics.py`, `preprocess.py`, `fc4.py`
- **Выполнить** задания в ноутбуке `lab.ipynb`.

## Окружение и проверка

```bash
uv sync          # создаёт .venv и ставит зависимости из pyproject.toml / uv.lock
uv run pytest    # запуск тестов
```
