
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch

from .infer import load_model, run_auralguard

try:
    from .audio_quality import audio_quality_report
except Exception:
    audio_quality_report = None

try:
    from .decision_gate import make_safe_report
except Exception:
    make_safe_report = None

try:
    from .explain_plus import beginner_friendly_explanation_plus
except Exception:
    beginner_friendly_explanation_plus = None


def parse_args():
    parser = argparse.ArgumentParser(description="Professional AuralGuard-AASIST++ forensic dashboard")
    parser.add_argument("--checkpoint", required=True, help="Final AuralGuard checkpoint")
    parser.add_argument("--aasist-root", default="external/aasist")
    parser.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--feature-dim", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def build_model_args(args):
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        aasist_root=args.aasist_root,
        aasist_config=args.aasist_config,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        feature_dim=args.feature_dim,
        device=args.device,
    )


def compute_quality(audio_path: str, sample_rate: int) -> dict:
    if audio_quality_report is None:
        return {"warnings": [], "metrics": {}}
    try:
        return audio_quality_report(audio_path, target_sr=sample_rate)
    except Exception as exc:
        return {
            "warnings": [f"Audio-quality diagnostics could not be computed: {exc}"],
            "metrics": {},
        }


def apply_safe_decision(report: dict, quality: dict) -> dict:
    if make_safe_report is None:
        out = dict(report)
        out["raw_decision"] = report.get("decision", "unknown")
        out["review_required"] = "suspicious" in str(report.get("decision", "")).lower()
        out["decision_gate"] = {"warnings": quality.get("warnings", [])}
        return out

    return make_safe_report(
        report,
        quality_warnings=quality.get("warnings", []),
        ood_warning=False,
    )


def make_explanation(report: dict) -> str:
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = str(report.get("attack_type", "unknown"))
    decision = str(report.get("decision", "unknown"))
    suspicious_segments = report.get("suspicious_segments", [])
    warnings = report.get("decision_gate", {}).get("warnings", [])

    if beginner_friendly_explanation_plus is not None:
        return beginner_friendly_explanation_plus(
            fake_prob=fake_prob,
            attack_type=attack_type,
            suspicious_segments=suspicious_segments,
            safe_decision=decision,
            warnings=warnings,
        )

    if "real" in decision.lower():
        return (
            "The system gives this audio a low fake score, so it is treated as likely real. "
            "The detected sound patterns are closer to real human speech than to generated speech."
        )
    if "fake" in decision.lower():
        return (
            "The system gives this audio a high fake score, so it is treated as likely fake. "
            "The detected sound patterns are similar to generated or manipulated speech."
        )
    return (
        "The system is not fully confident. The result should be reviewed by a human before making a final decision."
    )


def decision_style(decision: str) -> tuple[str, str]:
    d = (decision or "").lower()
    if "real" in d:
        return "real", "LIKELY REAL"
    if "fake" in d:
        return "fake", "LIKELY FAKE"
    return "review", "HUMAN REVIEW"


def main_decision_html(report: dict) -> str:
    decision = str(report.get("decision", "unknown"))
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = str(report.get("attack_type", "unknown"))
    review_required = bool(report.get("review_required", False))
    cls, label = decision_style(decision)

    return f"""
    <div class="decision-panel {cls}">
        <div class="panel-label">Final Forensic Assessment</div>
        <div class="decision-text">{label}</div>
        <div class="decision-description">
            This is the final decision from AuralGuard-AASIST++.
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <span>Fake Probability</span>
                <b>{fake_prob:.4f}</b>
            </div>
            <div class="metric-card">
                <span>Attack-Type Clue</span>
                <b>{html.escape(attack_type)}</b>
            </div>
            <div class="metric-card">
                <span>Human Review</span>
                <b>{"Required" if review_required else "Not Required"}</b>
            </div>
        </div>
    </div>
    """


def evidence_summary_html(report: dict) -> str:
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = str(report.get("attack_type", "unknown"))
    suspicious_segments = report.get("suspicious_segments", [])
    gate = report.get("decision_gate", {})
    warnings = gate.get("warnings", [])

    top_segment_text = "No suspicious timestamp region was highlighted."
    if suspicious_segments:
        top = max(suspicious_segments, key=lambda x: float(x.get("fake_score", 0.0)))
        top_segment_text = (
            f"{float(top.get('start', 0.0)):.1f}s–{float(top.get('end', 0.0)):.1f}s "
            f"(score {float(top.get('fake_score', 0.0)):.2f})"
        )

    warning_text = "No reliability warnings."
    if warnings:
        warning_text = "<br>".join([html.escape(str(w)) for w in warnings])

    return f"""
    <div class="evidence-panel">
        <h3>Evidence Summary</h3>
        <div class="evidence-row">
            <span>Model score</span>
            <b>{fake_prob:.4f}</b>
        </div>
        <div class="evidence-row">
            <span>Attack-type clue</span>
            <b>{html.escape(attack_type)}</b>
        </div>
        <div class="evidence-row">
            <span>Strongest suspicious region</span>
            <b>{top_segment_text}</b>
        </div>
        <div class="evidence-row">
            <span>Reliability notes</span>
            <b>{warning_text}</b>
        </div>
    </div>
    """


def quality_html(quality: dict) -> str:
    metrics = quality.get("metrics", {}) or {}
    warnings = quality.get("warnings", []) or []

    if metrics:
        metric_items = ""
        for key, value in metrics.items():
            label = html.escape(str(key).replace("_", " ").title())
            if isinstance(value, float):
                value_text = f"{value:.4f}"
            else:
                value_text = str(value)
            metric_items += f"""
            <div class="quality-item">
                <span>{label}</span>
                <b>{html.escape(value_text)}</b>
            </div>
            """
    else:
        metric_items = "<div class='empty-message'>No audio-quality metrics available.</div>"

    if warnings:
        warnings_html = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)
        warning_block = f"""
        <div class="warning-box">
            <b>Reliability Warnings</b>
            <ul>{warnings_html}</ul>
        </div>
        """
    else:
        warning_block = """
        <div class="ok-box">
            <b>No major audio-quality warnings detected.</b>
        </div>
        """

    return f"""
    <div class="quality-panel">
        <h3>Audio Quality Diagnostics</h3>
        <div class="quality-grid">{metric_items}</div>
        {warning_block}
    </div>
    """


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = load_model(build_model_args(args), device)

    def analyze(audio_path):
        if audio_path is None:
            empty = "<div class='empty-message'>Upload an audio file to begin analysis.</div>"
            return (
                empty,
                empty,
                "Upload an audio file to generate a simple explanation.",
                [],
                {},
                empty,
                "",
            )

        quality = compute_quality(audio_path, args.sample_rate)

        raw_report = run_auralguard(
            audio_path,
            model,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            device=device,
        )

        report = apply_safe_decision(raw_report, quality)
        report["beginner_explanation"] = make_explanation(report)
        report["quality_report"] = quality

        export = {
            "final_auralguard_result": report,
            "quality_report": quality,
        }

        report_path = Path("results") / "demo_professional_last_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(export, indent=2), encoding="utf-8")

        return (
            main_decision_html(report),
            evidence_summary_html(report),
            report.get("beginner_explanation", ""),
            report.get("suspicious_segments", []),
            report.get("evidence", {}),
            quality_html(quality),
            str(report_path),
        )

    css = """
    :root {
        --bg-dark: #07111f;
        --bg-mid: #0b1b32;
        --panel: rgba(14, 27, 49, 0.95);
        --panel-soft: rgba(22, 39, 68, 0.95);
        --line: rgba(131, 183, 255, 0.26);
        --white: #ffffff;
        --muted: #e6eefc;
        --blue: #4da3ff;
        --cyan: #28e0e8;
        --green: #39d98a;
        --yellow: #ffd166;
        --red: #ff5c7a;
    }

    body, .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(40, 224, 232, 0.16), transparent 30%),
            radial-gradient(circle at top right, rgba(77, 163, 255, 0.16), transparent 32%),
            linear-gradient(135deg, var(--bg-dark), var(--bg-mid)) !important;
        color: var(--white) !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }

    .gradio-container {
        max-width: 1180px !important;
    }

    .main-header {
        border: 1px solid var(--line);
        background: linear-gradient(135deg, rgba(17, 40, 72, 0.97), rgba(8, 17, 31, 0.98));
        border-radius: 22px;
        padding: 28px 30px;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.36);
        margin-bottom: 18px;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 13px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .main-title {
        font-size: 42px;
        line-height: 1.05;
        margin: 0;
        font-weight: 950;
        color: var(--white);
    }

    .subtitle {
        font-size: 16px;
        color: var(--white);
        margin-top: 12px;
        max-width: 820px;
        opacity: 0.92;
    }

    .feature-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .feature-chip {
        border: 1px solid rgba(77, 163, 255, 0.28);
        background: rgba(77, 163, 255, 0.12);
        color: var(--white);
        border-radius: 999px;
        padding: 7px 11px;
        font-size: 13px;
        font-weight: 800;
    }

    .input-panel {
        border: 1px solid var(--line) !important;
        background: var(--panel) !important;
        border-radius: 20px !important;
        box-shadow: 0 18px 45px rgba(0,0,0,0.24) !important;
        padding: 16px !important;
    }

    .decision-panel {
        border-radius: 24px;
        padding: 26px;
        border: 1px solid var(--line);
        background: linear-gradient(135deg, rgba(19, 38, 68, 0.98), rgba(8, 17, 31, 0.98));
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
    }

    .decision-panel.real { border-color: rgba(57, 217, 138, 0.65); }
    .decision-panel.fake { border-color: rgba(255, 92, 122, 0.68); }
    .decision-panel.review { border-color: rgba(255, 209, 102, 0.65); }

    .panel-label {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 950;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .decision-text {
        font-size: 48px;
        font-weight: 950;
        margin-top: 8px;
        letter-spacing: -0.03em;
        color: var(--white);
    }

    .decision-panel.real .decision-text { color: var(--green); }
    .decision-panel.fake .decision-text { color: var(--red); }
    .decision-panel.review .decision-text { color: var(--yellow); }

    .decision-description {
        color: var(--white);
        opacity: 0.92;
        margin-top: 8px;
        font-size: 15px;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 20px;
    }

    .metric-card, .quality-item, .evidence-row {
        background: rgba(255,255,255,0.065);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 14px;
    }

    .metric-card span, .quality-item span, .evidence-row span {
        display: block;
        color: var(--white);
        opacity: 0.88;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 850;
    }

    .metric-card b {
        display: block;
        margin-top: 7px;
        font-size: 25px;
        color: var(--white);
    }

    .evidence-panel, .quality-panel {
        border: 1px solid var(--line);
        background: var(--panel-soft);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.24);
        color: var(--white);
    }

    .evidence-panel h3, .quality-panel h3 {
        color: var(--white);
        margin-top: 0;
        margin-bottom: 16px;
        font-size: 23px;
    }

    .evidence-row {
        margin-bottom: 10px;
    }

    .evidence-row b {
        display: block;
        color: var(--white);
        margin-top: 6px;
        font-size: 17px;
        line-height: 1.4;
    }

    .quality-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 14px;
    }

    .quality-item b {
        display: block;
        margin-top: 5px;
        color: var(--white);
        font-size: 18px;
    }

    .warning-box {
        border: 1px solid rgba(255, 209, 102, 0.55);
        background: rgba(255, 209, 102, 0.12);
        border-radius: 14px;
        padding: 13px;
        color: var(--white);
    }

    .ok-box {
        border: 1px solid rgba(57, 217, 138, 0.45);
        background: rgba(57, 217, 138, 0.12);
        border-radius: 14px;
        padding: 13px;
        color: var(--white);
    }

    .empty-message {
        border: 1px dashed rgba(230, 238, 252, 0.42);
        border-radius: 18px;
        padding: 18px;
        color: var(--white);
        background: rgba(255,255,255,0.045);
    }

    textarea, input, .gradio-dropdown, .gradio-textbox {
        background: rgba(255,255,255,0.06) !important;
        color: var(--white) !important;
        border-color: rgba(131, 183, 255, 0.28) !important;
    }

    .prose, .markdown, label, .wrap, .json-holder, .tab-nav, .tabitem {
        color: var(--white) !important;
    }

    .gradio-container h1,
    .gradio-container h2,
    .gradio-container h3,
    .gradio-container p,
    .gradio-container span,
    .gradio-container label {
        color: var(--white) !important;
    }

        /* Make Evidence Packet, Suspicious Regions, and Simple Explanation readable */
    #evidence-json-box,
    #suspicious-json-box,
    #simple-explanation-box {
        background: #000000 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
    }

    #evidence-json-box *,
    #suspicious-json-box *,
    #simple-explanation-box * {
        background: #000000 !important;
        color: #ffffff !important;
    }

    #evidence-json-box pre,
    #suspicious-json-box pre,
    #evidence-json-box code,
    #suspicious-json-box code,
    #evidence-json-box .json-holder,
    #suspicious-json-box .json-holder {
        background: #000000 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

        #simple-explanation-box,
    #simple-explanation-box * {
        background: #000000 !important;
        color: #ffffff !important;
    }

    #simple-explanation-box textarea {
        background: #000000 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }

    #simple-explanation-box label {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    #evidence-json-box label,
    #suspicious-json-box label,
    #simple-explanation-box label {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    button.primary {
        background: linear-gradient(90deg, var(--blue), var(--cyan)) !important;
        color: #03101f !important;
        border: none !important;
        font-weight: 950 !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 35px rgba(40, 224, 232, 0.26) !important;
    }

    @media (max-width: 900px) {
        .metric-grid, .quality-grid {
            grid-template-columns: 1fr;
        }

        .decision-text {
            font-size: 38px;
        }
    }
    """

    with gr.Blocks(css=css, title="AuralGuard-AASIST++ Professional Dashboard", theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div class="main-header">
                <div class="eyebrow">Forensic Audio Analysis</div>
                <h1 class="main-title">AuralGuard-AASIST++</h1>
                <div class="subtitle">
                    Robust and explainable audio deepfake detection for synthetic speech,
                    accented English, and interview-style recordings.
                </div>
                <div class="feature-row">
                    <span class="feature-chip">Final Decision</span>
                    <span class="feature-chip">Evidence Summary</span>
                    <span class="feature-chip">Audio Quality Diagnostics</span>
                    <span class="feature-chip">Suspicious Timestamp Evidence</span>
                    <span class="feature-chip">Exportable Report</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["input-panel"]):
                gr.Markdown("### Upload Audio")
                audio_input = gr.Audio(type="filepath", label="Audio file")
                analyze_btn = gr.Button("Run Forensic Analysis", variant="primary")
                gr.Markdown(
                    """
                    The dashboard returns the final AuralGuard-AASIST++ decision, reliability notes,
                    and forensic evidence for review.
                    """
                )

            with gr.Column(scale=2):
                final_decision = gr.HTML()

        with gr.Tab("Evidence Summary"):
            evidence_summary = gr.HTML()

        with gr.Tab("Beginner Explanation"):
            beginner_explanation = gr.Textbox(
    label="Simple explanation",
    lines=8,
    elem_id="simple-explanation-box"
)

        with gr.Tab("Suspicious Timestamp Evidence"):
            suspicious_segments = gr.JSON(
    label="Suspicious timestamp regions",
    elem_id="suspicious-json-box"
)

        with gr.Tab("Evidence Packet"):
            evidence_packet = gr.JSON(
    label="Evidence packet",
    elem_id="evidence-json-box"
)

        with gr.Tab("Audio Quality Diagnostics"):
            quality_report = gr.HTML()

        with gr.Tab("Export"):
            export_path = gr.Textbox(label="Saved JSON report path")

        analyze_btn.click(
            fn=analyze,
            inputs=audio_input,
            outputs=[
                final_decision,
                evidence_summary,
                beginner_explanation,
                suspicious_segments,
                evidence_packet,
                quality_report,
                export_path,
            ],
        )

        gr.HTML(
            """
            <div style="margin-top: 18px; color: #ffffff; opacity: 0.85; font-size: 13px; text-align: center;">
                This dashboard provides forensic decision support, not legal proof.
                Uncertain, noisy, or out-of-domain audio should be reviewed by a human.
            </div>
            """
        )

    demo.launch(share=args.share, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
