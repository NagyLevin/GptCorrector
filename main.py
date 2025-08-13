from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import time
import os
from datetime import datetime

load_dotenv()

# --- Settings ---
FILES_DIR  = Path("/mnt/c/Users/Levinwork/Documents/Nytud/1feladat/celanyag/leiratok")
OUTPUT_DIR = Path("/mnt/c/Users/Levinwork/Documents/Nytud/1feladat/celanyag/javtest_out")

MODEL = "gpt-4o-mini"
SYSTEM_INSTRUCTIONS = (
    "Feladatod: a bemeneti szöveg helyesírási és nyelvtani javítása. "
    "Csak a hibákat javítsd (betűhibák, elütések, írásjelek, ragozás), "
    "a jelentést ne változtasd meg, ne adj hozzá és ne hagyj ki mondatokat. "
    "Formázást, sortöréseket tartsd meg. Válaszolj kizárólag a javított szöveggel."
)

VISITED_PATH = Path("./visited.txt")

# --- Chunking config (tune if needed) ---
CHUNK_CHARS = 12000         # aim for ~8–12k chars per chunk to avoid output caps
MAX_OUTPUT_TOKENS = 8000    # per chunk; safe headroom
TEMPERATURE = 0             # stable, deterministic edits

# FUNCTIONS
_start_time = None  # init timer state

def timer(action="start"):
    global _start_time
    if action == "start":
        _start_time = time.time()
        print("Timer started...")
    elif action == "stop":
        if _start_time is None:
            print("Start the timer first!")
        else:
            elapsed = time.time() - _start_time
            print(f"Elapsed: {elapsed:.3f} sec")
            _start_time = None
    else:
        print("Unknown timer action. Use: timer('start') or timer('stop')")

def say_time():
    now = datetime.now()
    print("Time now:", now.strftime("%H:%M:%S"))

def _ensure_visited_file():
    VISITED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not VISITED_PATH.exists():
        VISITED_PATH.write_text("", encoding="utf-8")

def check_and_add_visited(text: str) -> bool:
    """
    True -> it already exists do nothing
    False -> does not exist
    """
    _ensure_visited_file()
    with open(VISITED_PATH, "r", encoding="utf-8") as f:
        visited = {line.strip() for line in f if line.strip()}
    return text in visited

def add_to_visited(text: str) -> bool:
    """
       True -> it already exists do nothing
       False -> does not exist
    """
    _ensure_visited_file()
    with open(VISITED_PATH, "r", encoding="utf-8") as f:
        visited = {line.strip() for line in f if line.strip()}
    if text in visited:
        return True
    with open(VISITED_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    return False

# --- search for files: recursively, only .txt ---
def iter_txt_files(root: Path):
    # rglob + strict suffix check (case-insensitive)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".txt":
            yield p

# --- Chunk helper: prefer splitting on paragraph boundaries, with hard-split fallback ---
def chunk_by_paragraphs(s: str, max_chars: int = CHUNK_CHARS):
    if len(s) <= max_chars:
        return [s]
    parts, buf, size = [], [], 0
    for para in s.split("\n\n"):
        block = para + "\n\n"
        if size + len(block) > max_chars and buf:
            parts.append("".join(buf))
            buf, size = [block], len(block)
        else:
            buf.append(block)
            size += len(block)
    if buf:
        parts.append("".join(buf))
    # hard-split any oversize remainder
    final = []
    for c in parts:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars])
    return final

def main():
    if not FILES_DIR.exists():
        raise SystemExit(f"INPUT folder does not exist: {FILES_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY (.env).")

    client = OpenAI()

    txt_files = list(iter_txt_files(FILES_DIR))
    if not txt_files:
        print("No .txt files found (even recursively).")
        return

    print(f"{len(txt_files)} file(s) found (recursively).")

    for src in txt_files:
        # key for visited.txt: relative path from the input folder
        rel_key = str(src.relative_to(FILES_DIR)).replace("\\", "/")

        if check_and_add_visited(rel_key):
            print(f"SKIP (visited): {rel_key}")
            continue  # skip the whole process

        text = src.read_text(encoding="utf-8", errors="replace")

        # Skip empty files gracefully
        if not text.strip():
            print(f"SKIP (empty): {rel_key}")
            add_to_visited(rel_key)
            continue

        say_time()
        timer("start")

        # --- CHUNKED processing ---
        corrected_parts = []
        chunks = chunk_by_paragraphs(text, CHUNK_CHARS)
        print(f"Processing in {len(chunks)} chunk(s)...")

        for idx, part in enumerate(chunks, start=1):
            t0 = time.time()
            resp = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_INSTRUCTIONS,
                input=part.strip() if part.strip() else " ",
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )
            corrected_parts.append(resp.output_text)
            dt = time.time() - t0
            print(f"  chunk {idx}/{len(chunks)} done in {dt:.2f}s")

        corrected = "".join(corrected_parts)

        # mirror the folder structure in the OUTPUT directory
        dst = OUTPUT_DIR / src.relative_to(FILES_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(corrected, encoding="utf-8")

        # add to visited.txt only after successful processing
        add_to_visited(rel_key)
        timer("stop")

        print(f"OK: {rel_key} -> {dst}")

    print("Done.")

if __name__ == "__main__":
    main()
