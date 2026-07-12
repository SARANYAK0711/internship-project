"""
make_report.py
--------------
Generates the professional project report `MajorProject_YourName.pdf`
using ReportLab. Replace "Your Name" below with your actual name.
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
OUTPUT_PDF = os.path.join(BASE_DIR, "MajorProject_YourName.pdf")

STUDENT_NAME = "Your Name"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="MyTitle", fontSize=24, leading=30, alignment=TA_CENTER,
    textColor=colors.HexColor("#1e3a5f"), spaceAfter=12, fontName="Helvetica-Bold",
))
styles.add(ParagraphStyle(
    name="Subtitle", fontSize=13, leading=18, alignment=TA_CENTER,
    textColor=colors.HexColor("#4a4a4a"), spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="H2", fontSize=15, leading=20, textColor=colors.HexColor("#1e3a5f"),
    spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold",
))
styles.add(ParagraphStyle(
    name="Body2", fontSize=10.5, leading=16, alignment=TA_JUSTIFY, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="Code2", fontSize=9, leading=13, fontName="Courier",
    textColor=colors.HexColor("#2d2d2d"), backColor=colors.HexColor("#f2f2f2"),
    borderPadding=6, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="Bullet2", fontSize=10.5, leading=16, leftIndent=14, spaceAfter=3,
))


def h2(text):
    return Paragraph(text, styles["H2"])


def body(text):
    return Paragraph(text, styles["Body2"])


def bullet(text):
    return Paragraph(f"&bull;&nbsp;&nbsp;{text}", styles["Bullet2"])


def build():
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="AI-Based Disease Prediction System",
        author=STUDENT_NAME,
    )
    e = []  # story elements

    # ----- Title page -----
    e.append(Spacer(1, 4 * cm))
    e.append(Paragraph("AI-Based Disease Prediction System", styles["MyTitle"]))
    e.append(Paragraph("Using Machine Learning", styles["MyTitle"]))
    e.append(Spacer(1, 1 * cm))
    e.append(Paragraph("Major Project Report", styles["Subtitle"]))
    e.append(Spacer(1, 2 * cm))
    e.append(Paragraph(f"Submitted by: <b>{STUDENT_NAME}</b>", styles["Subtitle"]))
    e.append(Paragraph("Technology: Python &bull; Pandas &bull; NumPy &bull; Scikit-learn", styles["Subtitle"]))
    e.append(Paragraph("Model: Decision Tree Classifier", styles["Subtitle"]))
    e.append(PageBreak())

    # ----- Problem Statement -----
    e.append(h2("1. Problem Statement"))
    e.append(body(
        "Early prediction of diseases plays a critical role in timely diagnosis "
        "and treatment. Many patients delay seeking medical help because they are "
        "unsure whether their symptoms are serious. This project builds a machine "
        "learning application that predicts the most likely disease based on the "
        "symptoms entered by the user, encouraging early medical consultation."
    ))

    # ----- Objectives -----
    e.append(h2("2. Objectives"))
    for item in [
        "Load a symptom&ndash;disease dataset.",
        "Perform data preprocessing.",
        "Train a machine learning classification model (Decision Tree).",
        "Accept symptoms from the user.",
        "Predict the most likely disease.",
        "Display the prediction in a clear, user-friendly format.",
        "Handle invalid or missing input gracefully.",
    ]:
        e.append(bullet(item))

    # ----- Tools & Technologies -----
    e.append(h2("3. Tools &amp; Technologies"))
    tech_data = [
        ["Tool", "Purpose"],
        ["Python 3", "Core programming language"],
        ["Pandas", "Loading and preprocessing the dataset"],
        ["NumPy", "Numerical operations"],
        ["Scikit-learn", "Decision Tree model, train/test split, metrics"],
        ["Joblib", "Saving and loading the trained model (optional)"],
    ]
    table = Table(tech_data, colWidths=[5 * cm, 10 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    e.append(table)

    # ----- System Architecture -----
    e.append(h2("4. System Architecture"))
    e.append(body(
        "The system follows a simple, linear machine learning pipeline. Each stage "
        "feeds its output into the next:"
    ))
    e.append(Paragraph(
        "Dataset (CSV) &rarr; Load (Pandas) &rarr; Preprocess (X / y split) &rarr; "
        "Train Decision Tree &rarr; User Symptom Input &rarr; Feature Vector &rarr; "
        "Prediction &rarr; Display Result",
        styles["Code2"],
    ))

    # ----- Dataset Description -----
    e.append(h2("5. Dataset Description"))
    e.append(body(
        "The dataset (<b>disease_dataset.csv</b>) uses a binary symptom matrix. "
        "There are <b>20 symptom columns</b> (features) where 1 means the symptom is "
        "present and 0 means absent, and one <b>disease</b> column (target). It "
        "contains <b>10 diseases</b> with 60 samples each (600 rows total), including "
        "small realistic noise so the model learns robust patterns."
    ))
    e.append(bullet("<b>Symptoms:</b> fever, cough, fatigue, headache, sore_throat, "
                    "runny_nose, body_ache, chills, nausea, vomiting, diarrhea, "
                    "abdominal_pain, loss_of_taste, shortness_of_breath, joint_pain, "
                    "skin_rash, dizziness, chest_pain, sneezing, sweating."))
    e.append(bullet("<b>Diseases:</b> Common Cold, Influenza (Flu), COVID-19, Malaria, "
                    "Dengue, Typhoid, Gastroenteritis, Migraine, Pneumonia, Food Poisoning."))

    # ----- Data Preprocessing -----
    e.append(h2("6. Data Preprocessing"))
    e.append(body(
        "Preprocessing prepares the raw CSV for the model. Fully empty rows are "
        "dropped, any missing symptom values are filled with 0 (symptom absent), and "
        "all symptom columns are cast to integers. The data is then split into the "
        "feature matrix X (symptoms) and target vector y (disease)."
    ))
    e.append(Paragraph(
        "X = df[symptom_columns]<br/>y = df['disease']<br/>"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, "
        "random_state=42, stratify=y)",
        styles["Code2"],
    ))

    # ----- Machine Learning Model -----
    e.append(h2("7. Machine Learning Model"))
    e.append(body(
        "A <b>Decision Tree Classifier</b> (scikit-learn) is used as the primary "
        "model. A decision tree learns a series of yes/no questions on the symptoms "
        "(e.g. &lsquo;Is fever present?&rsquo;) and follows the branches to reach a "
        "predicted disease at a leaf node. It is easy to interpret, fast to train, "
        "and well suited to binary symptom features. The model uses the "
        "&lsquo;entropy&rsquo; criterion for information-gain based splits and "
        "achieves about <b>95% accuracy</b> on the held-out test set."
    ))
    e.append(Paragraph(
        "model = DecisionTreeClassifier(criterion='entropy', random_state=42)<br/>"
        "model.fit(X_train, y_train)",
        styles["Code2"],
    ))

    # ----- Code Explanation -----
    e.append(h2("8. Code Explanation"))
    for item in [
        "<b>load_dataset()</b> &mdash; reads the CSV with Pandas and validates it exists.",
        "<b>preprocess_data()</b> &mdash; cleans data and splits it into X (features) and y (target).",
        "<b>train_model()</b> &mdash; trains the Decision Tree and reports test accuracy; optionally saves the model with joblib.",
        "<b>parse_user_input()</b> &mdash; converts comma-separated symptom numbers/names into a binary feature vector and warns about invalid entries.",
        "<b>predict_disease()</b> &mdash; runs the model and returns the predicted disease plus confidence scores.",
        "<b>display_result()</b> &mdash; prints the prediction in a clear, formatted box.",
        "<b>interactive_mode()</b> &mdash; repeatedly accepts symptoms until the user quits.",
    ]:
        e.append(bullet(item))

    # ----- Screenshots -----
    e.append(PageBreak())
    e.append(h2("9. Screenshots"))
    shots = [
        ("01_training_and_menu.png", "Figure 1: Loading the dataset, training the model, and the symptom menu."),
        ("02_prediction_result.png", "Figure 2: A disease prediction with confidence score."),
        ("03_invalid_input_handling.png", "Figure 3: Graceful handling of invalid input."),
    ]
    for filename, caption in shots:
        path = os.path.join(SHOTS_DIR, filename)
        if os.path.exists(path):
            img = Image(path)
            img.drawWidth = 15 * cm
            img.drawHeight = 15 * cm * (img.imageHeight / img.imageWidth)
            e.append(img)
            e.append(Paragraph(caption, styles["Subtitle"]))
            e.append(Spacer(1, 0.4 * cm))

    # ----- Results -----
    e.append(PageBreak())
    e.append(h2("10. Results"))
    e.append(body(
        "The Decision Tree model achieved approximately <b>95% accuracy</b> on the "
        "test set and correctly classified canonical symptom combinations for all "
        "ten diseases. For example, the symptoms fever, cough, loss of taste and "
        "shortness of breath were correctly predicted as <b>COVID-19</b>, while "
        "headache, nausea and dizziness were predicted as <b>Migraine</b>."
    ))

    # ----- Conclusion -----
    e.append(h2("11. Conclusion"))
    e.append(body(
        "This project successfully demonstrates a complete machine learning workflow "
        "&mdash; from data loading and preprocessing to model training, prediction and "
        "result display &mdash; for predicting diseases from symptoms. The Decision "
        "Tree Classifier provides accurate, interpretable predictions and the "
        "application handles user input robustly. It is a solid, beginner-friendly "
        "foundation for understanding applied machine learning."
    ))

    # ----- Future Scope -----
    e.append(h2("12. Future Scope"))
    for item in [
        "Add more diseases and symptoms for broader coverage.",
        "Train on real, large-scale clinical datasets.",
        "Compare with other models such as Random Forest, Naive Bayes and SVM.",
        "Build a web or mobile front-end for wider accessibility.",
        "Incorporate symptom severity levels instead of simple yes/no values.",
        "Integrate with electronic health records for personalised predictions.",
    ]:
        e.append(bullet(item))

    doc.build(e)
    print(f"Report generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    build()
