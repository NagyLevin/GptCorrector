from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import time
import os
from datetime import datetime

load_dotenv()

# --- Beállítások ---
FILES_DIR  = Path("/mnt/c/Users/Levinwork/Documents/Nytud/1feladat/celanyag/javtest")
OUTPUT_DIR = Path("/mnt/c/Users/Levinwork/Documents/Nytud/1feladat/celanyag/javtest_out")

MODEL = "gpt-4o-mini"
SYSTEM_INSTRUCTIONS = (
    "Feladatod: a bemeneti szöveg helyesírási és nyelvtani javítása. "
    "Csak a hibákat javítsd (betűhibák, elütések, írásjelek, ragozás), "
    "a jelentést ne változtasd meg, ne adj hozzá és ne hagyj ki mondatokat. "
    "Formázást, sortöréseket tartsd meg. Válaszolj kizárólag a javított szöveggel."
)

VISITED_PATH = Path("./visited.txt")

# FV

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

# --- seatch for files: recursively, csak .txt ---
def iter_txt_files(root: Path):
    # rglob + szigorú szuffix ellenőrzés (kis/nagybetűk miatt)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".txt":
            yield p

def main():
    if not FILES_DIR.exists():
        raise SystemExit(f"INPUT mappa nem létezik: {FILES_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Hiányzik az OPENAI_API_KEY (.env).")

    client = OpenAI()

    txt_files = list(iter_txt_files(FILES_DIR))
    if not txt_files:
        print("Nincs .txt fájl (rekurzívan sem).")
        return

    print(f"{len(txt_files)} fájl találat (rekurzívan).")

    for src in txt_files:
        # kulcs a visited-hez: a bemeneti mappához viszonyított relatív útvonal
        rel_key = str(src.relative_to(FILES_DIR)).replace("\\", "/")

        if check_and_add_visited(rel_key):
            print(f"SKIP (visited): {rel_key}")
            continue

        text = src.read_text(encoding="utf-8", errors="replace")

        # stateless hívás: minden fájlhoz új kérés, mindig megkapja a SYSTEM_INSTRUCTIONS-t
        resp = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=text.strip() if text.strip() else " "  # üres fájl ellen
        )
        say_time()
        timer("start")

        corrected = resp.output_text

        # OUTPUT mappában tükrözzük a struktúrát
        dst = OUTPUT_DIR / src.relative_to(FILES_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(corrected, encoding="utf-8")

        # csak SIKER után jegyezzük fel, hogy feldolgoztuk
        add_to_visited(rel_key)
        timer("stop")

        print(f"OK: {rel_key} -> {dst}")

    print("Kész.")

if __name__ == "__main__":
    main()
