from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import time
import os
from datetime import datetime

load_dotenv()  # GPT API key is in a .env file

# --- Settings ---
FILES_DIR  = Path("/home/szabol/leiratok")
OUTPUT_DIR = Path("/home/szabol/podcast_corrected_with_gpt")

"""
GPT modell and instructions
in the future might update to GPT5
"""
MODEL = "gpt-4o-mini"
SYSTEM_INSTRUCTIONS = (
    "Feladatod: a bemeneti szöveg helyesírási és nyelvtani javítása. "
    "Csak a hibákat javítsd (betűhibák, elütések, írásjelek, ragozás), "
    "a jelentést ne változtasd meg, ne adj hozzá és ne hagyj ki mondatokat. "
    "Formázást, sortöréseket tartsd meg. Válaszolj kizárólag a javított szöveggel."
)

VISITED_PATH  = Path("./visited.txt")   # names of processed files (relative keys)
TIMEOUTS_PATH = Path("./timeouts.txt")  # names of aborted files (relative keys)

# --- Chunking config (tune if needed) ---
CHUNK_CHARS = 12000         # aim ~8–12k chars per chunk
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

"""
say current time
"""

def say_time():
    now = datetime.now()
    print("Time now:", now.strftime("%H:%M:%S"))


def _ensure_file(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")

def _ensure_visited_file():
    _ensure_file(VISITED_PATH)

def _ensure_timeouts_file():
    _ensure_file(TIMEOUTS_PATH)

def check_and_add_visited(text: str) -> bool:
    """
    True -> already exists in visited (caller should skip)
    False -> not present
    """
    _ensure_visited_file()
    with open(VISITED_PATH, "r", encoding="utf-8") as f:
        visited = {line.strip() for line in f if line.strip()}
    return text in visited

def add_to_visited(text: str) -> bool:
    """
    True -> already existed (so nothing appended)
    False -> newly added
    """
    _ensure_visited_file()
    with open(VISITED_PATH, "r", encoding="utf-8") as f:
        visited = {line.strip() for line in f if line.strip()}
    if text in visited:
        return True
    with open(VISITED_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    return False

def add_to_timeouts(text: str) -> None:
    """
    Append rel_key to timeouts.txt if not already there.
    Called when a file aborts due to a missing/failed chunk.
    """
    _ensure_timeouts_file()
    with open(TIMEOUTS_PATH, "r", encoding="utf-8") as f:
        cur = {line.strip() for line in f if line.strip()}
    if text not in cur:
        with open(TIMEOUTS_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")

"""
check for txt files
"""

def iter_txt_files(root: Path):
    # recursively yield only .txt files
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
        # rel_key is the visited/timeout key
        rel_key = str(src.relative_to(FILES_DIR)).replace("\\", "/")
        print(f"Working on: {src.name}")

        if check_and_add_visited(rel_key):
            print(f"SKIP (visited): {rel_key}")
            continue

        text = src.read_text(encoding="utf-8", errors="replace")

        # Skip empty files gracefully (still mark visited)
        if not text.strip():
            print(f"SKIP (empty): {rel_key}")
            add_to_visited(rel_key)
            continue

        say_time()
        timer("start")

        # --- ALL-OR-NOTHING CHUNKED processing ---
        chunks = chunk_by_paragraphs(text, CHUNK_CHARS)
        print(f"Processing in {len(chunks)} chunk(s)...")

        corrected_parts = []
        try:
            for idx, part in enumerate(chunks, start=1):
                t0 = time.time()
                # Any exception here will abort the file and go to timeouts
                resp = client.responses.create(
                    model=MODEL,
                    instructions=SYSTEM_INSTRUCTIONS,
                    input=part.strip() if part.strip() else " ",
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=TEMPERATURE,
                )
                out = getattr(resp, "output_text", "") or ""
                if not out.strip():
                    raise RuntimeError(f"Empty output for chunk {idx}/{len(chunks)}")
                corrected_parts.append(out)
                dt = time.time() - t0
                print(f"  chunk {idx}/{len(chunks)} done in {dt:.2f}s")

            corrected = "".join(corrected_parts)

            # mirror the folder structure in the OUTPUT directory
            dst = OUTPUT_DIR / src.relative_to(FILES_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(corrected, encoding="utf-8")

            # success: mark visited only AFTER successful full processing
            add_to_visited(rel_key)
            timer("stop")
            print(f"OK: {rel_key} -> {dst}")

        except Exception as e:
            # abort this file, do NOT mark visited, DO add to timeouts
            timer("stop")
            print(f"[ABORT FILE] {rel_key} due to chunk failure: {e}")
            add_to_timeouts(rel_key)
            # move on to next file
            continue

    print("Done.")

if __name__ == "__main__":
    while True:  # periodically scan for new files
        main()
        print("Waiting for new files to process")
        say_time()
        time.sleep(60)
