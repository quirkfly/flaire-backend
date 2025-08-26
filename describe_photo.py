#!/usr/bin/env python3
import os
import sys
import base64
import mimetypes
import argparse
import json
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class Config:
    model: str = "gpt-5"  # or "gpt-5-chat-latest"
    default_prompt: str = (
        "You are given up to five photos of the same person.\n"
        "Step 1: Briefly describe each photo in 2–3 sentences.\n"
        "Step 2: Summarize consistent themes (style, hobbies, vibe).\n"
        "Step 3: Using ONLY visible cues (no guessing sensitive attributes like age, ethnicity, religion, health, or occupation), "
        "write a single short dating-profile blurb that feels warm, genuine, and confident.\n\n"
        "Profile guidelines:\n"
        "- Base hobbies/interests only on things clearly seen (e.g. guitar, bike, hiking gear).\n"
        "- Avoid age, profession, politics, religion, location unless explicitly shown.\n"
        "- No demographic labels.\n"
        "- 80–140 words for the profile.\n"
        "- Provide 2–3 playful conversation starters relevant to the context.\n"
        "Return JSON with keys: descriptions (list of strings), themes, profile, starters (list of strings)."
    )

def b64_data_url(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"Not an image (or unknown type): {path}")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def make_prompt(user_prompt: str, name: str | None, app: str | None, tone: str | None) -> str:
    extras = []
    if name:
        extras.append(f"Use the first name '{name}' once in the profile.")
    if app:
        extras.append(f"Style it to fit the tone of {app}.")
    if tone:
        extras.append(f"Overall tone: {tone}.")
    addon = ("\n" + "\n".join(extras)) if extras else ""
    return user_prompt + addon

def analyze_photos(image_paths: list[str], cfg: Config, name: str | None, app: str | None, tone: str | None):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    client = OpenAI(api_key=api_key)

    contents = [{"type": "input_text", "text": make_prompt(cfg.default_prompt, name, app, tone)}]
    for p in image_paths:
        contents.append({"type": "input_image", "image_url": b64_data_url(p)})

    resp = client.responses.create(
        model=cfg.model,
        input=[{
            "role": "user",
            "content": contents,
        }],
    )

    text = resp.output_text
    try:
        data = json.loads(text)
        print("\n=== Photo Descriptions ===")
        for i, desc in enumerate(data.get("descriptions", []), 1):
            print(f"Photo {i}: {desc}")
        print("\n=== Themes ===\n" + data.get("themes", "").strip())
        print("\n=== Dating Profile ===\n" + data.get("profile", "").strip())
        starters = data.get("starters", [])
        if starters:
            print("\n=== Conversation Starters ===")
            for i, s in enumerate(starters, 1):
                print(f"{i}. {s}")
    except Exception:
        print(text)

def parse_args():
    p = argparse.ArgumentParser(description="Analyze up to 5 photos and generate a dating-site profile.")
    p.add_argument("image_paths", nargs="+", help="Paths to 1–5 image files (png/jpg/jpeg/webp).")
    p.add_argument("--name", help="First name to optionally include once in the profile.", default=None)
    p.add_argument("--app", help="Target app/site style (e.g., Tinder, Hinge, Bumble).", default=None)
    p.add_argument("--tone", help="Tone hint (e.g., playful, classy, witty, wholesome).", default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if len(args.image_paths) > 5:
        print("Error: Please provide no more than 5 photos.", file=sys.stderr)
        sys.exit(1)
    try:
        analyze_photos(args.image_paths, Config(), args.name, args.app, args.tone)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
