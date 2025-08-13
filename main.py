from dotenv import load_dotenv
import os
from pathlib import Path
from openai import OpenAI

load_dotenv()

FILES_DIR  = Path("/mnt/c/Users/Levinwork/Documents/Nytud/1feladat/celanyag/javtest")
OUTPUT_DIR = Path("/mnt/c/Users/Levinwork/Documents/Nytud/1feladat/celanyag/javitva")

MODEL = "gpt-4o-mini"
SYSTEM_INSTRUCTIONS = (
    "Feladatod: a bemeneti szöveg helyesírási és nyelvtani javítása. "
    "Csak a hibákat javítsd (betűhibák, elütések, írásjelek, ragozás), "
    "a jelentést ne változtasd meg, ne adj hozzá és ne hagyj ki mondatokat. "
    "Formázást, sortöréseket tartsd meg. Válaszolj kizárólag a javított szöveggel."
)

def main():
    if not FILES_DIR.exists():
        raise SystemExit(f"INPUT mappa nem létezik: {FILES_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY in .env file.")

    client = OpenAI()

    txt_files = sorted([p for p in FILES_DIR.glob("*.txt") if p.is_file()])
    if not txt_files:
        print("Nincs .txt fájl a bemeneti mappában.")
        return

    print(f"{len(txt_files)} fájl feldolgozása...")

    for src in txt_files:
        text = src.read_text(encoding="utf-8", errors="replace")

        resp = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,  # <-- itt a javítás
            input=text,
        )
        corrected = resp.output_text

        dst = OUTPUT_DIR / src.name
        dst.write_text(corrected, encoding="utf-8")
        print(f"OK: {src.name} -> {dst}")

    print("Kész.")

if __name__ == "__main__":
    main()
