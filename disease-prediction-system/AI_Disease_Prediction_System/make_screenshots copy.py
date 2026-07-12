"""
make_screenshots.py
-------------------
Renders sample terminal output of the disease prediction system into
PNG images, saved in the `screenshots/` folder. This is only a helper to
produce documentation assets; it is not part of the core application.
"""

import os
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SHOTS_DIR, exist_ok=True)

# Terminal-like colour scheme.
BG = (30, 30, 46)
FG = (205, 214, 244)
GREEN = (166, 227, 161)
YELLOW = (249, 226, 175)
PROMPT_BAR = (49, 50, 68)

PADDING = 24
LINE_HEIGHT = 22


def load_font(size=16):
    """Try a few common monospaced fonts, fall back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = load_font(15)


def render(lines, filename, title="Terminal"):
    """Render a list of (text, colour) tuples into a PNG image."""
    width = 820
    height = PADDING * 2 + LINE_HEIGHT * (len(lines) + 1)
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # Title bar with traffic-light dots.
    draw.rectangle([0, 0, width, 30], fill=PROMPT_BAR)
    for i, colour in enumerate([(243, 139, 168), (249, 226, 175), (166, 227, 161)]):
        draw.ellipse([14 + i * 20, 10, 26 + i * 20, 22], fill=colour)
    draw.text((width // 2 - 40, 8), title, font=FONT, fill=FG)

    y = 40
    for text, colour in lines:
        draw.text((PADDING, y), text, font=FONT, fill=colour)
        y += LINE_HEIGHT

    path = os.path.join(SHOTS_DIR, filename)
    img.save(path)
    print(f"Saved {path}")


def main():
    # Screenshot 1: training + menu.
    render(
        [
            ("$ python disease_prediction.py", GREEN),
            ("*******************************************************", FG),
            ("   AI-BASED DISEASE PREDICTION SYSTEM (Decision Tree)", FG),
            ("*******************************************************", FG),
            ("[INFO] Dataset loaded successfully: 600 rows, 21 columns.", YELLOW),
            ("[INFO] Preprocessing complete: 20 symptom features.", YELLOW),
            ("[INFO] Model trained. Test accuracy: 95.00%", YELLOW),
            ("", FG),
            ("Available symptoms:", FG),
            ("--------------------------------------------------", FG),
            ("   1. Fever            2. Cough            3. Fatigue", FG),
            ("   4. Headache         5. Sore Throat      6. Runny Nose", FG),
            ("   7. Body Ache        8. Chills           9. Nausea", FG),
            ("  10. Vomiting        11. Diarrhea        12. Abdominal Pain", FG),
            ("  13. Loss Of Taste   14. Shortness Of Breath ...", FG),
            ("--------------------------------------------------", FG),
        ],
        "01_training_and_menu.png",
        title="disease_prediction.py",
    )

    # Screenshot 2: a prediction result.
    render(
        [
            ("Your symptoms: fever, cough, loss_of_taste, shortness_of_breath", GREEN),
            ("", FG),
            ("=======================================================", FG),
            ("               DISEASE PREDICTION RESULT", FG),
            ("=======================================================", FG),
            ("  Symptoms entered : Fever, Cough, Loss Of Taste, Shortness Of Breath", FG),
            ("  Predicted disease: >>> COVID-19 <<<", GREEN),
            ("-------------------------------------------------------", FG),
            ("  Top possibilities (confidence):", FG),
            ("    - COVID-19             100.0%  ####################", YELLOW),
            ("=======================================================", FG),
            ("  NOTE: This is an ML-based prediction for educational", FG),
            ("        purposes only. Please consult a qualified doctor.", FG),
            ("=======================================================", FG),
        ],
        "02_prediction_result.png",
        title="disease_prediction.py",
    )

    # Screenshot 3: invalid input handling.
    render(
        [
            ("Your symptoms: xyz, 999", GREEN),
            ("[WARN] Ignored unrecognised entries: xyz, 999", YELLOW),
            ("[ERROR] None of the entered symptoms were valid. Please try again.", (243, 139, 168)),
            ("", FG),
            ("Your symptoms: headache, nausea, dizziness", GREEN),
            ("", FG),
            ("  Symptoms entered : Headache, Nausea, Dizziness", FG),
            ("  Predicted disease: >>> Migraine <<<", GREEN),
            ("    - Migraine             100.0%  ####################", YELLOW),
        ],
        "03_invalid_input_handling.png",
        title="disease_prediction.py",
    )


if __name__ == "__main__":
    main()
