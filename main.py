from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import io
import json
import math
import os
import re
import socket
import time
import uuid
from typing import List, Optional

import boto3
import docx
import fitz  # PyMuPDF
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fpdf import FPDF
from google import genai
from google.genai import types
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="StrataDocs AI — Enterprise Document Workspace")

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
s3_client = boto3.client("s3", region_name=AWS_REGION)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

FAST_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]


def run_with_hard_timeout(fn, timeout_sec: int, *args, **kwargs):
    """Runs blocking network calls with a strict timeout without waiting on stalled threads."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_sec)
    except FuturesTimeout:
        raise TimeoutError(f"Upstream Gemini API timed out after {timeout_sec}s.")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def ndjson(payload: dict) -> str:
    return json.dumps(payload) + "\n"


def format_clean_error(err: Exception) -> str:
    msg = str(err)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        match = re.search(r"retry in (\d+)", msg, re.IGNORECASE)
        sec = match.group(1) if match else "45"
        return f"Gemini free-tier quota is temporarily full. Please wait ~{sec}s and retry."
    return msg[:260]


def get_db_conn():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "docworkspace"),
        user=os.getenv("DB_USER", "admin"),
        password=os.getenv("DB_PASSWORD"),
    )
    register_vector(conn)
    return conn


@app.on_event("startup")
def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            s3_key TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector(768)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


def normalize_vec(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def get_embedding(text: str) -> List[float]:
    def _call():
        res = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return normalize_vec(res.embeddings[0].values)

    return run_with_hard_timeout(_call, 15)


def chunk_text(text: str, base_size: int = 1200, overlap: int = 150) -> List[str]:
    size = 2000 if len(text) > 50000 else base_size
    effective_overlap = min(overlap, size // 5)
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - effective_overlap
    cleaned = [c.strip() for c in chunks if c.strip()]
    if len(cleaned) > 35:
        step = len(cleaned) / 35.0
        cleaned = [cleaned[int(i * step)] for i in range(35)]
    return cleaned


def generate_fast(contents, system_instruction: Optional[str] = None, max_tokens: int = 750):
    """Low-latency generation with thinking disabled, 503 retry backoff, and model failover."""
    global FAST_MODELS
    last_err = None

    for model_name in list(FAST_MODELS):
        for attempt in range(2):
            def _call(m=model_name):
                try:
                    cfg = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=max_tokens,
                        temperature=0.3,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                    return gemini_client.models.generate_content(
                        model=m, contents=contents, config=cfg
                    )
                except Exception as inner_e:
                    if "thinking" in str(inner_e).lower():
                        cfg = types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            max_output_tokens=max_tokens,
                            temperature=0.3,
                        )
                        return gemini_client.models.generate_content(
                            model=m, contents=contents, config=cfg
                        )
                    raise

            try:
                res = run_with_hard_timeout(_call, 22)
                if FAST_MODELS[0] != model_name:
                    FAST_MODELS = [model_name] + [m for m in FAST_MODELS if m != model_name]
                return res
            except Exception as e:
                last_err = e
                err_str = str(e)
                if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt == 0:
                    time.sleep(2.0)
                    continue
                break

    raise last_err


def transcribe_image_with_vision(image_bytes: bytes, mime_type: str) -> str:
    response = generate_fast(
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Transcribe all text, handwritten notes, tables, and formulas accurately in clean Markdown.",
        ]
    )
    return response.text or ""


def extract_universal_text(file_bytes: bytes, filename: str, content_type: str) -> str:
    ext = filename.lower().split(".")[-1]

    if ext in ["png", "jpg", "jpeg", "webp"]:
        mime = content_type if content_type.startswith("image/") else f"image/{ext}"
        return transcribe_image_with_vision(file_bytes, mime)

    if ext == "pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        ocr_pages_used = 0
        try:
            for page_num, page in enumerate(doc):
                page_text = page.get_text().strip()
                if len(page_text) >= 40:
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                elif ocr_pages_used < 10:
                    pix = page.get_pixmap(dpi=130)
                    ocr_text = transcribe_image_with_vision(pix.tobytes("png"), "image/png")
                    text_parts.append(f"--- Page {page_num + 1} (Vision OCR) ---\n{ocr_text}")
                    ocr_pages_used += 1
                elif page_text:
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        finally:
            doc.close()
        return "\n\n".join(text_parts)

    if ext == "docx":
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    return file_bytes.decode("utf-8", errors="replace")


def upload_output_and_presign(data_bytes: bytes, filename: str, content_type: str) -> str:
    key = f"outputs/{uuid.uuid4().hex[:8]}_{filename}"
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=data_bytes,
        ContentType=content_type,
    )
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=300,
    )


def sanitize_latin1(text: str) -> str:
    replacements = {"₹": "Rs. ", "—": "-", "–": "-", "•": "*", "’": "'", "“": '"', "”": '"'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_bytes(title: str, body_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, sanitize_latin1(title))
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    clean_body = re.sub(r"[#*`]+", "", body_text)
    pdf.multi_cell(0, 7, sanitize_latin1(clean_body))
    return bytes(pdf.output())


@app.get("/health")
def health_check():
    return {"status": "ok", "instance": socket.gethostname()}


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/documents")
def list_documents():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, filename, s3_key, created_at FROM documents ORDER BY id DESC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"id": r[0], "filename": r[1], "s3_key": r[2], "created_at": str(r[3])}
        for r in rows
    ]


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    filename = file.filename or "document"
    content_type = file.content_type or "application/octet-stream"

    def event_stream():
        try:
            yield ndjson({"status": "progress", "message": "Uploading raw document to Amazon S3..."})
            s3_key = f"uploads/{uuid.uuid4().hex[:8]}_{filename}"
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=file_bytes,
                ContentType=content_type,
            )

            ext = filename.lower().split(".")[-1]
            if ext == "pdf":
                tmp_doc = fitz.open(stream=file_bytes, filetype="pdf")
                page_count = len(tmp_doc)
                tmp_doc.close()
                yield ndjson({"status": "progress", "message": f"Extracting text from {page_count}-page PDF..."})
            elif ext in ["png", "jpg", "jpeg", "webp"]:
                yield ndjson({"status": "progress", "message": "Running Vision LLM handwriting & OCR transcription..."})
            else:
                yield ndjson({"status": "progress", "message": "Extracting document text..."})

            extracted = extract_universal_text(file_bytes, filename, content_type)
            if not extracted.strip():
                yield ndjson({"status": "error", "detail": "Could not extract text from file."})
                return

            chunks = chunk_text(extracted)
            batch_size = 20
            total_batches = math.ceil(len(chunks) / batch_size)
            all_embeddings = []

            for b_idx in range(total_batches):
                batch = chunks[b_idx * batch_size : (b_idx + 1) * batch_size]
                for attempt in range(2):
                    yield ndjson({
                        "status": "progress",
                        "message": f"Generating embeddings (Batch {b_idx + 1}/{total_batches} • {len(chunks)} chunks)...",
                    })
                    try:
                        def _embed(b=batch):
                            return gemini_client.models.embed_content(
                                model="gemini-embedding-001",
                                contents=b,
                                config=types.EmbedContentConfig(output_dimensionality=768),
                            )

                        res = run_with_hard_timeout(_embed, 20)
                        for emb in res.embeddings:
                            all_embeddings.append(normalize_vec(emb.values))
                        break
                    except Exception as e:
                        err_str = str(e)
                        if ("429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt == 0:
                            match = re.search(r"retry in (\d+)", err_str, re.IGNORECASE)
                            wait_sec = min((int(match.group(1)) + 2) if match else 45, 55)
                            for remaining in range(wait_sec, 0, -1):
                                yield ndjson({
                                    "status": "backoff",
                                    "message": f"Free-tier quota paused on Batch {b_idx + 1}/{total_batches}. Auto-resuming in {remaining}s...",
                                })
                                time.sleep(1)
                            continue
                        raise

            yield ndjson({
                "status": "progress",
                "message": f"Saving {len(chunks)} vectors to PostgreSQL (pgvector on EBS)...",
            })
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO documents (filename, s3_key, extracted_text) VALUES (%s, %s, %s) RETURNING id;",
                (filename, s3_key, extracted),
            )
            doc_id = cur.fetchone()[0]

            records = [(doc_id, filename, chunk, emb) for chunk, emb in zip(chunks, all_embeddings)]
            execute_values(
                cur,
                "INSERT INTO document_chunks (doc_id, filename, chunk_text, embedding) VALUES %s;",
                records,
            )

            conn.commit()
            cur.close()
            conn.close()

            yield ndjson({
                "status": "done",
                "doc_id": doc_id,
                "filename": filename,
                "s3_key": s3_key,
                "chunks_indexed": len(chunks),
            })
        except Exception as e:
            yield ndjson({"status": "error", "detail": format_clean_error(e)})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


class AskRequest(BaseModel):
    prompt: str
    doc_id: Optional[int] = None


@app.post("/ask")
def ask_or_command(req: AskRequest):
    try:
        query_emb = get_embedding(req.prompt)
        vec_literal = str(query_emb)

        conn = get_db_conn()
        cur = conn.cursor()
        if req.doc_id is not None:
            cur.execute(
                """
                SELECT filename, chunk_text
                FROM document_chunks
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 4;
                """,
                (req.doc_id, vec_literal),
            )
        else:
            cur.execute(
                """
                SELECT filename, chunk_text
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 4;
                """,
                (vec_literal,),
            )
        matches = cur.fetchall()
        cur.close()
        conn.close()

        context = "\n\n".join([f"[File: {m[0]}]\n{m[1]}" for m in matches])
        if not context:
            context = "No matching document content found."

        prompt_lower = req.prompt.lower()
        wants_csv = any(k in prompt_lower for k in ["csv", "spreadsheet", "excel"])
        wants_pdf = any(k in prompt_lower for k in ["pdf", "study guide", "generate report", "pdf report"])

        generated_downloads = []

        if wants_csv:
            sys_prompt = (
                "You are a fast data extraction engine. Based strictly on the document context below, "
                "output ONLY valid CSV rows (up to 15 concise rows; first line = headers, subsequent lines = data). "
                "Quote fields containing commas. Do NOT include markdown fences (```) or extra commentary.\n\n"
                f"CONTEXT:\n{context}"
            )
            response = generate_fast(req.prompt, system_instruction=sys_prompt, max_tokens=750)
            csv_text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", (response.text or "").strip())
            url = upload_output_and_presign(csv_text.encode("utf-8"), "extracted_export.csv", "text/csv")
            generated_downloads.append({"filename": "extracted_export.csv", "url": url})
            preview_lines = "\n".join(csv_text.splitlines()[:10])
            final_answer = (
                f"Generated **extracted_export.csv** ({len(csv_text.splitlines())} rows) and uploaded it to Amazon S3.\n\n"
                f"```csv\n{preview_lines}\n```"
            )
        elif wants_pdf:
            sys_prompt = (
                "You are StrataDocs AI. Write a crisp, well-structured executive report or study guide "
                "(3–4 short sections with bullet points) based strictly on the document context below.\n\n"
                f"CONTEXT:\n{context}"
            )
            response = generate_fast(req.prompt, system_instruction=sys_prompt, max_tokens=750)
            report_text = (response.text or "No content generated.").strip()
            pdf_bytes = build_pdf_bytes("StrataDocs AI — Document Report", report_text)
            url = upload_output_and_presign(pdf_bytes, "stratadocs_report.pdf", "application/pdf")
            generated_downloads.append({"filename": "stratadocs_report.pdf", "url": url})
            final_answer = report_text
        else:
            sys_prompt = (
                "You are StrataDocs AI, an Enterprise Document Assistant. Answer the user's question "
                "concisely and accurately using clean Markdown based strictly on the context below.\n\n"
                f"CONTEXT:\n{context}"
            )
            response = generate_fast(req.prompt, system_instruction=sys_prompt, max_tokens=650)
            final_answer = (response.text or "No response returned.").strip()

        return {
            "answer": final_answer,
            "downloads": generated_downloads,
            "served_by_instance": socket.gethostname(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=format_clean_error(e))