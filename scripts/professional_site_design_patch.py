from pathlib import Path
import re

TARGET = Path("src/demo_professional.py")

PROFESSIONAL_HEADER = """gr.HTML(\"\"\"
            <section class="hero-header">
                <div class="hero-grid">
                    <div class="hero-copy">
                        <div class="eyebrow">Forensic Audio Analysis</div>
                        <h1 class="main-title">AuralGuard-AASIST++</h1>
                        <p class="subtitle">
                            Robust audio deepfake detection with accent-aware false-alarm reduction,
                            human-review support, and readable forensic evidence.
                        </p>
                    </div>
                    <div class="hero-status-card">
                        <div class="status-label">System Mode</div>
                        <div class="status-value">Decision Support</div>
                        <div class="status-note">Not legal proof • Review uncertain cases</div>
                    </div>
                </div>

                <div class="workflow-strip">
                    <div class="workflow-step">
                        <span class="step-number">01</span>
                        <span class="step-title">Upload audio</span>
                    </div>
                    <div class="workflow-step">
                        <span class="step-number">02</span>
                        <span class="step-title">Analyze acoustic evidence</span>
                    </div>
                    <div class="workflow-step">
                        <span class="step-number">03</span>
                        <span class="step-title">Review decision and report</span>
                    </div>
                </div>
            </section>
        \"\"\")"""

CSS_ADD = """
/* PROFESSIONAL SITE HEADER REDESIGN
   Removes the bubble/chip look and replaces it with a structured product-style header. */

.hero-header {
    border: 1px solid rgba(148, 163, 184, 0.22);
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(8, 20, 38, 0.98));
    border-radius: 18px;
    padding: 30px 34px 24px 34px;
    box-shadow: 0 22px 60px rgba(0, 0, 0, 0.28);
    margin-bottom: 22px;
}

.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    gap: 26px;
    align-items: center;
}

.hero-copy {
    max-width: 820px;
}

.eyebrow {
    color: #38e8f2 !important;
    font-size: 12px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 900;
    margin-bottom: 12px;
}

.main-title {
    font-size: 46px;
    line-height: 1.04;
    margin: 0;
    font-weight: 950;
    letter-spacing: -0.04em;
    color: #ffffff !important;
}

.subtitle {
    font-size: 17px;
    line-height: 1.55;
    color: #dbeafe !important;
    margin-top: 14px;
    max-width: 800px;
}

.hero-status-card {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(96, 165, 250, 0.28);
    border-radius: 16px;
    padding: 18px 18px;
    text-align: left;
}

.status-label {
    color: #93c5fd !important;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 850;
}

.status-value {
    color: #ffffff !important;
    font-size: 22px;
    font-weight: 900;
    margin-top: 6px;
}

.status-note {
    color: #cbd5e1 !important;
    font-size: 13px;
    line-height: 1.45;
    margin-top: 8px;
}

.workflow-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0;
    margin-top: 26px;
    border-top: 1px solid rgba(148, 163, 184, 0.18);
    padding-top: 18px;
}

.workflow-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-right: 18px;
}

.step-number {
    color: #38e8f2 !important;
    font-size: 12px;
    font-weight: 950;
    letter-spacing: 0.10em;
}

.step-title {
    color: #f8fafc !important;
    font-size: 14px;
    font-weight: 800;
}

/* Remove old bubble-chip style if any remains */
.feature-row,
.feature-chip {
    display: none !important;
}

/* Make tabs cleaner and less button-like */
.gradio-container button[role="tab"],
.gradio-container .tab-nav button {
    background: transparent !important;
    border: none !important;
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    padding: 12px 14px !important;
    font-weight: 650 !important;
    border-radius: 0 !important;
}

.gradio-container button[role="tab"][aria-selected="true"],
.gradio-container .tab-nav button.selected {
    background: rgba(255, 255, 255, 0.96) !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    border-radius: 4px 4px 0 0 !important;
}

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }

    .workflow-strip {
        grid-template-columns: 1fr;
        gap: 12px;
    }

    .main-title {
        font-size: 36px;
    }
}
"""

def replace_header(text: str) -> str:
    pattern = re.compile(
        r'gr\.HTML\("""\s*<div class="main-header">.*?</div>\s*"""\)',
        flags=re.DOTALL
    )
    new_text, n = pattern.subn(PROFESSIONAL_HEADER, text, count=1)

    if n:
        print("Replaced old main-header with professional header.")
        return new_text

    marker = "        with gr.Row():"
    if marker in text and "hero-header" not in text:
        text = text.replace(marker, "        " + PROFESSIONAL_HEADER + "\n\n" + marker, 1)
        print("Inserted professional header using fallback.")
        return text

    print("Could not find old header, or professional header already exists.")
    return text

def add_css(text: str) -> str:
    if "PROFESSIONAL SITE HEADER REDESIGN" in text:
        print("Professional CSS already present.")
        return text

    css_pos = text.find('css = """')
    if css_pos == -1:
        print("Could not find css block.")
        return text

    end_css = text.find('"""', css_pos + len('css = """'))
    if end_css == -1:
        print("Could not find end of css block.")
        return text

    text = text[:end_css] + "\n" + CSS_ADD + "\n" + text[end_css:]
    print("Added professional header CSS.")
    return text

def main():
    if not TARGET.exists():
        raise SystemExit("Could not find src/demo_professional.py. Run this from the project root.")

    original = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(".py.before_professional_site_design")
    backup.write_text(original, encoding="utf-8")

    text = original
    text = replace_header(text)
    text = add_css(text)

    TARGET.write_text(text, encoding="utf-8")
    print(f"Done. Backup saved as: {backup}")
    print("Now test with:")
    print(r"python -m py_compile src\demo_professional.py")

if __name__ == "__main__":
    main()
