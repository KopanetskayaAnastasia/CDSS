# -*- coding: utf-8 -*-
"""
Улучшенная RAG-система поддержки врачебных решений
- Выбор области диагностики
- История диалога с навигацией
- Источники с названием PDF
- Уровень доверенности
- LLM строго по фрагментам
"""

import os
import glob
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd

# Загрузка переменных окружения
from dotenv import load_dotenv
load_dotenv()

# PDF обработка
import fitz

# GigaChat
from gigachat import GigaChat

# RAG компоненты
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# ====================== КОНФИГУРАЦИЯ ======================
DOCS_FOLDER = "docs"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 5
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Области диагностики (для фильтрации)
DIAGNOSTIC_AREAS = {
    "Все области": None,
    "Кардиология": ["гипертензия", "сердечная", "хсн", "артериальная", "давление", "сердце", "инфаркт"],
    "Эндокринология": ["диабет", "сахарный", "глюкоза", "инсулин", "щитовидная"],
    "Пульмонология": ["пневмония", "одышка", "кашель", "легочный", "хобл", "дыхание"],
    "Неврология": ["инсульт", "головная", "мигрень", "невролог", "парез"],
    "Гастроэнтерология": ["печень", "желудок", "гастрит", "панкреатит", "кишечник"]
}

# Читаем секреты из .env
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

if not GIGACHAT_CREDENTIALS:
    st.error("⚠️ Ошибка: не найден ключ GigaChat!\n"
             "Создайте файл .env в корне проекта и добавьте:\n"
             "GIGACHAT_CREDENTIALS=ваш_ключ")
    st.stop()


# ====================== ЗАГРУЗКА PDF ======================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает текст из PDF с сохранением структуры"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        blocks = page.get_text("dict")
        for block in blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                    text += "\n"
            else:
                text += block.get("text", "") + "\n"
        text += "\n---\n"
    doc.close()
    return text


def load_all_documents(folder_path: str) -> Dict[str, str]:
    """Загружает все PDF из папки"""
    documents = {}
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return documents

    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        text = extract_text_from_pdf(pdf_file)
        if len(text.strip()) > 100:
            documents[filename] = text
    return documents


# ====================== ВЕКТОРНАЯ БАЗА ======================
def build_vector_store(documents: Dict[str, str], embedder, client, area_filter: str = None):
    """Создаёт векторную базу с фильтрацией по области"""
    try:
        client.delete_collection("clinical_guidelines")
    except:
        pass

    collection = client.create_collection("clinical_guidelines")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "---\n", "\n", r"\. ", " ", ""]
    )

    all_chunks = []
    all_metadatas = []
    all_ids = []

    chunk_counter = 0
    for filename, text in documents.items():
        # Определяем область документа (если есть фильтр)
        doc_area = detect_document_area(filename.lower())
        if area_filter and area_filter != "Все области" and doc_area != area_filter:
            continue

        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": filename,
                "text": chunk,
                "chunk_id": chunk_counter,
                "area": doc_area
            })
            all_ids.append(f"{filename}_{chunk_counter}")
            chunk_counter += 1

    if not all_chunks:
        return collection, 0

    # Генерация эмбеддингов
    progress_bar = st.progress(0, text="🔢 Вычисление эмбеддингов...")
    batch_size = 32
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_metadatas = all_metadatas[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]

        embeddings = embedder.encode(batch_chunks).tolist()
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            documents=batch_chunks
        )

        progress = min(1.0, (i + batch_size) / len(all_chunks))
        progress_bar.progress(progress)

    progress_bar.empty()
    return collection, len(all_chunks)


def detect_document_area(filename: str) -> str:
    """Определяет область медицины по имени файла"""
    for area, keywords in DIAGNOSTIC_AREAS.items():
        if area == "Все области":
            continue
        if keywords:
            for kw in keywords:
                if kw in filename.lower():
                    return area
    return "Другое"


def retrieve(query: str, collection, embedder, top_k: int = TOP_K_RESULTS) -> tuple[List[Dict], float]:
    """Расширенный поиск с возвратом уверенности"""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )

    retrieved = []
    if results['metadatas'] and results['metadatas'][0]:
        distances = results['distances'][0] if results['distances'] else []
        for i, meta in enumerate(results['metadatas'][0]):
            # Вычисляем уверенность (чем меньше расстояние, тем выше уверенность)
            confidence = max(0, min(100, int((1 - distances[i] / 2) * 100))) if distances else 50
            retrieved.append({
                "text": results['documents'][0][i] if results['documents'] else meta.get("text", ""),
                "source": meta.get("source", "unknown"),
                "area": meta.get("area", "Другое"),
                "confidence": confidence
            })

    # Средняя уверенность по всем найденным фрагментам
    avg_confidence = sum(r["confidence"] for r in retrieved) / len(retrieved) if retrieved else 0
    return retrieved, avg_confidence


# ====================== GIGACHAT ======================
def init_gigachat():
    return GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        scope=GIGACHAT_SCOPE,
        verify_ssl_certs=False,
        timeout=90,
        max_retries=3
    )


def build_medical_prompt(query: str, context_chunks: List[Dict]) -> str:
    """Формирует промпт с явными источниками и требованием строго по фрагментам"""
    context_text = ""
    sources_list = []

    for i, chunk in enumerate(context_chunks, 1):
        # Показываем название файла, а не просто "Документ"
        source_name = chunk['source'].replace('.pdf', '')
        context_text += f"\n[ИСТОЧНИК {i}: {source_name} (уверенность: {chunk['confidence']}%)]\n{chunk['text']}\n"
        sources_list.append(f"{i}. **{source_name}** (уверенность {chunk['confidence']}%)")

    sources_str = "\n".join(sources_list)

    prompt = f"""Ты — врач-эксперт, система поддержки принятия врачебных решений.

=== ГЛАВНОЕ ПРАВИЛО ===
Ты ОБЯЗАН отвечать ТОЛЬКО на основе приведённых ниже фрагментов клинических рекомендаций.
Если информации в фрагментах нет — НЕ выдумывай, а честно напиши: "В предоставленных рекомендациях эта информация не найдена".
Ни в коем случае не используй свои знания — только то, что написано в фрагментах.

=== КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ (ЕДИНСТВЕННЫЙ ИСТОЧНИК) ===
{context_text}

=== ВОПРОС ВРАЧА ===
{query}

=== ТВОЙ ОТВЕТ (только по фрагментам, со ссылками на источники) ===
"""
    return prompt, sources_str


def ask_gigachat(query: str, context_chunks: List[Dict]) -> Dict:
    """Отправляет запрос к GigaChat"""
    if not context_chunks:
        return {
            "answer": "📋 В загруженных клинических рекомендациях не найдено информации по вашему вопросу.\n\nРекомендации:\n• Уточните формулировку вопроса\n• Добавьте больше PDF-файлов",
            "sources": "Информация не найдена",
            "full_response": None,
            "confidence": 0
        }

    prompt, sources = build_medical_prompt(query, context_chunks)

    try:
        with init_gigachat() as giga:
            response = giga.chat(prompt)

        # Извлекаем уверенность из контекста
        avg_confidence = sum(c["confidence"] for c in context_chunks) / len(context_chunks)

        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
            "full_response": response,
            "confidence": avg_confidence
        }
    except Exception as e:
        return {
            "answer": f"❌ Ошибка: {str(e)}",
            "sources": sources,
            "full_response": None,
            "confidence": 0
        }


# ====================== ГЛАВНЫЙ КЛАСС ======================
class MedicalRAGSystem:
    def __init__(self):
        with st.spinner("🔄 Загрузка модели эмбеддингов..."):
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = None
        self.is_ready = False
        self.conversation_history = []
        self.documents_list = []
        self.total_chunks = 0
        self.current_area = "Все области"

        # ===== АВТОМАТИЧЕСКАЯ ЗАГРУЗКА =====
        try:
            self.collection = self.client.get_collection("clinical_guidelines")
            if self.collection.count() > 0:
                self.is_ready = True
                # Восстанавливаем список документов из метаданных
                results = self.collection.get()
                unique_sources = set()
                if results['metadatas']:
                    for meta in results['metadatas']:
                        if meta and 'source' in meta:
                            unique_sources.add(meta['source'])
                self.documents_list = list(unique_sources)
                self.total_chunks = self.collection.count()
        except:
            pass  # Базы нет — будет создана при первом нажатии

    def initialize(self, docs_folder: str = DOCS_FOLDER, area: str = "Все области") -> bool:
        self.current_area = area
        documents = load_all_documents(docs_folder)
        if not documents:
            return False

        self.documents_list = list(documents.keys())
        self.collection, self.total_chunks = build_vector_store(documents, self.embedder, self.client, area)
        self.is_ready = True
        return True

    def ask(self, query: str) -> Dict:
        if not self.is_ready or not self.collection:
            return {"answer": "⚠️ Система не инициализирована. Нажмите 'Загрузить PDF' один раз.",
                    "sources": "", "context": [], "confidence": 0}

        context_chunks, avg_confidence = retrieve(query, self.collection, self.embedder)
        result = ask_gigachat(query, context_chunks)
        result["context"] = context_chunks
        result["confidence"] = avg_confidence

        self.conversation_history.append({
            "id": len(self.conversation_history),
            "question": query,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        return result

    def clear_history(self):
        self.conversation_history = []

    def get_history_df(self) -> pd.DataFrame:
        if not self.conversation_history:
            return pd.DataFrame()
        return pd.DataFrame(self.conversation_history)

# ====================== ИНТЕРФЕЙС ======================
def run_ui():
    st.set_page_config(
        page_title="СППВР - Медицинская RAG система",
        page_icon="🩺",
        layout="wide"
    )

    # CSS для улучшения внешнего вида
    st.markdown("""
    <style>
    .stButton button { border-radius: 20px; }
    .confidence-high { color: #00ff88; font-weight: bold; }
    .confidence-mid { color: #ffaa00; font-weight: bold; }
    .confidence-low { color: #ff4444; font-weight: bold; }
    .source-badge { background-color: #2e7d64; padding: 2px 8px; border-radius: 15px; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🩺 Система поддержки врачебных решений (СППВР)")
    st.caption("RAG-архитектура + GigaChat | Основано на клинических рекомендациях Минздрава РФ")

    # Инициализация
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = MedicalRAGSystem()
        st.session_state.system_ready = False

    # ====================== БОКОВАЯ ПАНЕЛЬ ======================
    with st.sidebar:
        st.header("⚙️ Настройки")

        # Выбор области диагностики
        selected_area = st.selectbox("🏥 Область диагностики", list(DIAGNOSTIC_AREAS.keys()))

        # Загрузка базы
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Загрузить PDF", type="primary", use_container_width=True):
                with st.spinner("Загрузка и индексация документов..."):
                    st.session_state.system_ready = st.session_state.rag_system.initialize(
                        docs_folder=DOCS_FOLDER,
                        area=selected_area
                    )
                    if st.session_state.system_ready:
                        st.success(f"✅ Загружено {len(st.session_state.rag_system.documents_list)} документов")

        with col2:
            if st.button("🗑️ Очистить историю", use_container_width=True):
                st.session_state.rag_system.clear_history()
                st.success("История очищена")

        if st.session_state.system_ready:
            st.success(f"✅ Система готова")
            st.metric("📄 Документов", len(st.session_state.rag_system.documents_list))
            st.metric("🧩 Фрагментов", st.session_state.rag_system.total_chunks)

            with st.expander("📁 Загруженные документы"):
                for doc in st.session_state.rag_system.documents_list:
                    st.caption(f"📄 {doc}")
        else:
            st.info("📁 Положите PDF в папку `docs` и нажмите 'Загрузить PDF'")

    # ====================== ОСНОВНАЯ ОБЛАСТЬ ======================
    # Две колонки: диалог (слева) и история (справа)
    col_main, col_history = st.columns([2, 1])

    with col_main:
        # Поле ввода
        query = st.text_area("💬 Введите вопрос врачу:",
                             height=100,
                             placeholder="Пример: Какие препараты первой линии при артериальной гипертензии?")

        if st.button("🔍 Получить ответ", type="primary", use_container_width=True):
            if not st.session_state.system_ready:
                st.warning("⚠️ Сначала загрузите клинические рекомендации (кнопка слева)")
            elif query.strip():
                with st.spinner("🔬 Анализ клинических рекомендаций..."):
                    result = st.session_state.rag_system.ask(query)

                    # Отображение ответа
                    st.markdown("---")
                    st.markdown("### 💡 Ответ системы")

                    # Уровень доверенности
                    conf = result.get("confidence", 0)
                    if conf >= 70:
                        st.markdown(f"<span class='confidence-high'>🎯 Уровень доверенности: {conf:.0f}%</span>", unsafe_allow_html=True)
                    elif conf >= 40:
                        st.markdown(f"<span class='confidence-mid'>⚠️ Уровень доверенности: {conf:.0f}%</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='confidence-low'>❓ Уровень доверенности: {conf:.0f}% (рекомендуется проверить)</span>", unsafe_allow_html=True)

                    st.success(result["answer"])

                    # Источники
                    if result.get("sources") and result["sources"] != "Информация не найдена":
                        with st.expander("📄 Источники информации (PDF)"):
                            st.markdown(result["sources"])

                    # Детали поиска
                    with st.expander("🔬 Найденные фрагменты рекомендаций"):
                        for i, chunk in enumerate(result.get("context", []), 1):
                            source_name = chunk['source'].replace('.pdf', '')
                            conf_c = chunk.get('confidence', 0)
                            st.markdown(f"**Фрагмент {i}** — *{source_name}* (уверенность {conf_c}%)")
                            st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                            st.markdown("---")
            else:
                st.warning("Введите вопрос")

    with col_history:
        st.markdown("### 📜 История диалога")

        if st.session_state.rag_system.conversation_history:
            # Выбор вопроса из истории
            history_options = [
                f"{entry['timestamp']} - {entry['question'][:40]}..."
                for entry in st.session_state.rag_system.conversation_history
            ]
            selected_idx = st.selectbox(
                "Перейти к вопросу",
                options=range(len(history_options)),
                format_func=lambda x: history_options[x],
                key="history_selector"
            )

            if selected_idx is not None:
                entry = st.session_state.rag_system.conversation_history[selected_idx]
                with st.expander(f"❓ {entry['question'][:60]}", expanded=True):
                    st.caption(f"🕐 {entry['timestamp']}")
                    st.caption(f"🎯 Уверенность: {entry['confidence']:.0f}%")
                    st.markdown(f"**Ответ:** {entry['answer'][:300]}...")

            # Общая статистика
            st.markdown("---")
            avg_conf = sum(e['confidence'] for e in st.session_state.rag_system.conversation_history) / len(st.session_state.rag_system.conversation_history)
            st.metric("📊 Средняя уверенность", f"{avg_conf:.0f}%")

            # Кнопка экспорта
            if st.button("📎 Экспорт истории (JSON)", use_container_width=True):
                history_data = st.session_state.rag_system.get_history_df()
                st.download_button(
                    label="Скачать",
                    data=history_data.to_json(force_ascii=False, indent=2),
                    file_name=f"consultation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("Пока нет сохранённых диалогов. Задайте вопрос, и он появится здесь.")


if __name__ == "__main__":
    run_ui()