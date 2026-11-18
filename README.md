# RAG по финансовой отчётности

Александр Болгов

[Открыть эксперимент LangSmith](https://smith.langchain.com/public/94c16c9e-8fb2-4b67-9556-e43128aa1d5d/d) · [Скачать результаты Excel](test_results.xlsx)

## Результаты на 141 вопросе test

<!-- RESULTS -->

| Метрика | Среднее |
| --- | ---: |
| context_recall | 0.814066 |
| context_precision | 0.263095 |
| answer_correctness | 0.702128 |
| time | 7.633348 |

<!-- /RESULTS -->

99 из 141 ответа засчитаны. Время указано в секундах. Повторный прогон уложился в допуск ±5% по всем четырём метрикам.

## Как работает

PyMuPDF извлекает текст по страницам PDF, включая OCR сканов. Поиск сочетает BM25 и TF-IDF с SVD (scikit-learn и SciPy). Модель `deepseek-flash` отвечает по найденным фрагментам и при необходимости запрашивает дополнительный поиск. Excel формируется через openpyxl.

## Как запустить

Нужен Python 3.10+. Укажите курсовой `LLM_API_KEY` в `.env` по [примеру](.env.example).

```bash
python run_homework.py
python run_inference.py --repeat
```

Первая команда устанавливает зависимости, скачивает PDF и строит индекс. Вторая сохраняет новый Excel в `artifacts/test_repeat.xlsx` и сверяет его со сданным файлом. После прерывания добавьте `--resume` ко второй команде.
