"""
setup_keys.py
-------------
Creates the .env file by asking for the API keys in the terminal, so no text
editor is needed. Run with:  python -m app.setup_keys
"""

from app.config import BASE_DIR


def main():
    print("\nPaste your API keys below (right-click pastes in the Windows terminal).\n")
    gemini = ""
    while not gemini:
        gemini = input("Gemini API key (https://aistudio.google.com/apikey): ").strip()
    hf = input("Hugging Face token (https://huggingface.co/settings/tokens): ").strip()

    env_path = BASE_DIR / ".env"
    env_path.write_text(f"GEMINI_API_KEY={gemini}\nHF_API_KEY={hf}\n", encoding="utf-8")
    print(f"\nSaved {env_path.name}. You can change the keys later by running this again.\n")


if __name__ == "__main__":
    main()
