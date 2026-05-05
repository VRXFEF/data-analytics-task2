import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

# --- Настройки ---
MODEL = "llama-3.3-70b-versatile"
ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = Path(__file__).resolve().parent

INPUT_PATH  = TASK_DIR / "input.csv"
OUTPUT_PATH = TASK_DIR / "output.csv"
LOG_PATH    = TASK_DIR / "pipeline.log"


class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Тональность: positive/negative/neutral"
    )
    topic:   str = Field(description="Тема 2–5 слов по-русски")
    summary: str = Field(description="Краткое резюме одним предложением по-русски")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_api_key() -> str:
    load_dotenv(ROOT_DIR / ".env")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "Не найден GROQ_API_KEY. Добавьте в .env файл в корне проекта: GROQ_API_KEY=..."
        )
    return key


def build_prompt(review: str) -> str:
    return (
        "Проанализируй отзыв и верни СТРОГО один JSON-объект без дополнительного текста.\n"
        "Формат строго такой:\n"
        "{\"sentiment\": \"positive/negative/neutral\", \"topic\": \"...\", \"summary\": \"...\"}\n"
        "Правила:\n"
        "- sentiment строго одно из: positive, negative, neutral\n"
        "- topic: 2–5 слов на русском (тема отзыва)\n"
        "- summary: 1 предложение на русском\n"
        "Только JSON, никакого markdown, никаких пояснений.\n\n"
        f"Отзыв:\n{review}"
    )


def extract_json(text: str) -> dict | None:
    text = text.strip()
    # Убираем markdown-блоки если модель всё же обернула
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def normalize_result(obj: dict | None) -> dict:
    if not isinstance(obj, dict):
        return {"sentiment": None, "topic": None, "summary": None}

    sentiment = obj.get("sentiment")
    topic     = obj.get("topic")
    summary   = obj.get("summary")

    if sentiment not in {"positive", "negative", "neutral"}:
        sentiment = None
    topic   = topic.strip()   if isinstance(topic, str)   and topic.strip()   else None
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else None

    return {"sentiment": sentiment, "topic": topic, "summary": summary}


def call_groq(client: Groq, review: str) -> str:
    """Отправляет отзыв в Groq и возвращает сырой текст ответа."""
    backoff = 1.0
    for attempt in range(1, 5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — помощник по анализу отзывов. "
                            "Отвечай ТОЛЬКО JSON-объектом без markdown и пояснений."
                        ),
                    },
                    {"role": "user", "content": build_prompt(review)},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            return resp.choices[0].message.content or ""

        except Exception as e:
            msg = str(e)
            # 429 — rate limit, ретраим с задержкой
            if "429" in msg and attempt < 4:
                logging.warning("Rate limit (попытка %s/4), жду %.0fs...", attempt, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            logging.error("Ошибка Groq (попытка %s/4): %s", attempt, e)
            break
    return ""


def main() -> None:
    setup_logging()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    if not {"id", "review"}.issubset(set(df.columns)):
        raise ValueError(f"В input.csv нужны колонки 'id' и 'review', получено: {list(df.columns)}")

    client = Groq(api_key=load_api_key())
    results: list[dict] = []

    for row in df.itertuples(index=False):
        row_id = getattr(row, "id")
        review = getattr(row, "review")
        logging.info("Обрабатываю id=%s", row_id)

        try:
            raw  = call_groq(client, str(review))
            obj  = extract_json(raw)
            norm = normalize_result(obj)

            if all(v is None for v in norm.values()):
                logging.warning("Не удалось распарсить JSON для id=%s. Ответ: %r", row_id, raw)

            results.append({"id": row_id, **norm})

        except Exception as e:
            logging.exception("Ошибка при обработке id=%s: %s", row_id, e)
            results.append({"id": row_id, "sentiment": None, "topic": None, "summary": None})

    out_df = pd.DataFrame(results, columns=["id", "sentiment", "topic", "summary"])
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    logging.info("Готово! Результат сохранён в %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
