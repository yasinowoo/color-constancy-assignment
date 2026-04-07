# color-constancy-assignment

Домашнее задание посвящено двум этапам конвейера обработки изображений: балансу белого (Automatic White Balance) и преобразованию цветовых координат (Color Space Transform).

## Данные

1. Скачайте архив с данными:   **https://disk.360.yandex.ru/d/wufAja1zV2HWYA**
2. Распакуйте в корне репозитория

## Что нужно сделать

- **Реализовать** заготовки в `awb.py`, `cst.py`, `metrics.py`, `preprocess.py`
- **Выполнить** задания в ноутбуке `lab.ipynb`.

## Окружение и проверка

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```
