
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
    p = argparse.ArgumentParser(description="Clear AuralGuard-AASIST++ forensic dashboard demo")
    p.add_argument("--baseline-checkpoint", default="", help="Optional baseline checkpoint")
    p.add_argument("--final-checkpoint", required=True, help="Final AuralGuard checkpoint")
    p.add_argument("--baseline-name", default="Baseline AASIST")
    p.add_argument("--final-name", default="AuralGuard-AASIST++")
    p.add_argument("--aasist-root", default="external/aasist")
    p.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration-sec", type=float, default=4.0)
    p.add_argument("--feature-dim", type=int, default=160)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def load_checkpoint(args, checkpoint: str, device):
    model_args = SimpleNamespace(
        checkpoint=checkpoint,
        aasist_root=args.aasist_root,
        aasist_config=args.aasist_config,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        feature_dim=args.feature_dim,
        device=str(device),
    )
    return load_model(model_args, device)


def safe_quality_report(audio_path: str, sample_rate: int) -> dict:
    if audio_quality_report is None:
        return {"warnings": [], "metrics": {}}
    try:
        return audio_quality_report(audio_path, target_sr=sample_rate)
    except Exception as exc:
        return {"warnings": [f"Could not compute audio quality report: {exc}"], "metrics": {}}


def safe_make_report(raw_report: dict, quality: dict) -> dict:
    if make_safe_report is None:
        out = dict(raw_report)
        out["raw_decision"] = raw_report.get("decision", "unknown")
        out["review_required"] = "suspicious" in str(out.get("decision", "")).lower()
        out["decision_gate"] = {"warnings": quality.get("warnings", [])}
        return out

    return make_safe_report(
        raw_report,
        quality_warnings=quality.get("warnings", []),
        ood_warning=False,
    )


def simple_explanation(report: dict) -> str:
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = str(report.get("attack_type", "unknown"))
    decision = str(report.get("decision", "unknown"))
    warnings = report.get("decision_gate", {}).get("warnings", [])
    suspicious_segments = report.get("suspicious_segments", [])

    if beginner_friendly_explanation_plus is not None:
        return beginner_friendly_explanation_plus(
            fake_prob=fake_prob,
            attack_type=attack_type,
            suspicious_segments=suspicious_segments,
            safe_decision=decision,
            warnings=warnings,
        )

    if "real" in decision.lower():
        return "The final model gives this audio a low fake score, so it is treated as likely real."
    if "fake" in decision.lower():
        return "The final model gives this audio a high fake score, so it is treated as likely fake."
    return "The model is not fully confident, so this audio should be reviewed by a human."


def status_class(decision: str) -> tuple[str, str]:
    d = (decision or "").lower()
    if "real" in d:
        return "real", "LIKELY REAL"
    if "fake" in d:
        return "fake", "LIKELY FAKE"
    return "review", "HUMAN REVIEW"


def final_decision_banner(report: dict, model_name: str) -> str:
    decision = str(report.get("decision", "unknown"))
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = str(report.get("attack_type", "unknown"))
    review = bool(report.get("review_required", False))
    cls, label = status_class(decision)

    return f"""
    <div class="final-banner {cls}">
        <div class="final-label">FINAL AURALGUARD RESULT</div>
        <div class="final-decision">{label}</div>
        <div class="final-subtitle">This is the main decision from the improved final model.</div>
        <div class="final-grid">
            <div>
                <span>Fake probability</span>
                <b>{fake_prob:.4f}</b>
            </div>
            <div>
                <span>Attack-type clue</span>
                <b>{html.escape(attack_type)}</b>
            </div>
            <div>
                <span>Human review</span>
                <b>{"Yes" if review else "No"}</b>
            </div>
        </div>
    </div>
    """


def html_card(model_name: str, report: dict, subtitle: str, is_final: bool = False) -> str:
    decision = str(report.get("decision", "unknown"))
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = str(report.get("attack_type", "unknown"))
    review = bool(report.get("review_required", False))
    cls, label = status_class(decision)
    bar_width = max(0.0, min(100.0, fake_prob * 100.0))

    if is_final:
        tag = "USE THIS RESULT"
        role = "Final improved model"
    else:
        tag = "COMPARISON ONLY"
        role = "Older baseline model"

    return f"""
    <div class="result-card {cls} {'final-card' if is_final else 'baseline-card'}">
        <div class="card-topline">
            <div>
                <span class="role-tag {'final-tag' if is_final else 'baseline-tag'}">{tag}</span>
                <div class="model-name">{html.escape(model_name)}</div>
            </div>
            <span class="status-pill {cls}">{label}</span>
        </div>

        <div class="card-subtitle">{html.escape(role)} · {html.escape(subtitle)}</div>

        <div class="probability-row">
            <div>
                <div class="metric-label">Fake probability</div>
                <div class="probability-value">{fake_prob:.4f}</div>
            </div>
            <div>
                <div class="metric-label">Attack-type clue</div>
                <div class="attack-value">{html.escape(attack_type)}</div>
            </div>
        </div>

        <div class="score-track">
            <div class="score-fill {cls}" style="width: {bar_width:.1f}%"></div>
        </div>

        <div class="card-footer">
            <span>Decision: <b>{html.escape(decision)}</b></span>
            <span>Human review: <b>{"Yes" if review else "No"}</b></span>
        </div>
    </div>
    """


def comparison_panel(baseline_report: dict, final_report: dict) -> str:
    base_prob = float(baseline_report.get("fake_probability", 0.0))
    final_prob = float(final_report.get("fake_probability", 0.0))
    diff = base_prob - final_prob

    base_decision = baseline_report.get("decision", "unknown")
    final_decision = final_report.get("decision", "unknown")

    return f"""
    <div class="comparison-card">
        <div class="comparison-badge">MODEL COMPARISON</div>
        <h3>Baseline vs Final AuralGuard</h3>
        <p>
            The baseline is shown only to demonstrate how the older model behaves.
            The final AuralGuard-AASIST++ result is the main result to report.
        </p>
        <div class="comparison-grid">
            <div>
                <span class="small-label">Baseline decision</span>
                <b>{html.escape(str(base_decision))}</b>
            </div>
            <div>
                <span class="small-label">Final decision</span>
                <b>{html.escape(str(final_decision))}</b>
            </div>
            <div>
                <span class="small-label">Fake probability change</span>
                <b>{diff:.4f}</b>
            </div>
        </div>
    </div>
    """


def quality_panel(quality: dict) -> str:
    metrics = quality.get("metrics", {}) or {}
    warnings = quality.get("warnings", []) or []

    metric_html = ""
    for key, value in metrics.items():
        label = html.escape(str(key).replace("_", " ").title())
        val = f"{value:.4f}" if isinstance(value, float) else str(value)
        metric_html += f"<div class='quality-metric'><span>{label}</span><b>{html.escape(val)}</b></div>"

    if not metric_html:
        metric_html = "<p>No audio-quality metrics available.</p>"

    if warnings:
        warnings_html = "".join([f"<li>{html.escape(str(w))}</li>" for w in warnings])
        warning_block = f"""
        <div class="warning-box">
            <b>Reliability warnings</b>
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
    <div class="quality-card">
        <h3>Audio Quality Diagnostics</h3>
        <div class="quality-grid">{metric_html}</div>
        {warning_block}
    </div>
    """


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    final_model = load_checkpoint(args, args.final_checkpoint, device)
    baseline_model = None
    if args.baseline_checkpoint:
        baseline_model = load_checkpoint(args, args.baseline_checkpoint, device)

    def analyze(audio_path):
        if audio_path is None:
            empty = "<div class='empty-state'>Upload an audio file to begin analysis.</div>"
            return empty, empty, empty, empty, "Upload an audio file to generate an explanation.", [], {}, empty, ""

        quality = safe_quality_report(audio_path, args.sample_rate)

        raw_final = run_auralguard(
            audio_path,
            final_model,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            device=device,
        )
        final_report = safe_make_report(raw_final, quality)
        final_report["beginner_explanation"] = simple_explanation(final_report)
        final_report["quality_report"] = quality

        if baseline_model is not None:
            raw_base = run_auralguard(
                audio_path,
                baseline_model,
                sample_rate=args.sample_rate,
                duration_sec=args.duration_sec,
                device=device,
            )
            baseline_report = safe_make_report(raw_base, quality)
        else:
            baseline_report = {
                "decision": "not loaded",
                "fake_probability": 0.0,
                "attack_type": "not loaded",
                "review_required": False,
                "suspicious_segments": [],
                "evidence": {},
            }

        combined = {
            "baseline_comparison_only": baseline_report,
            "final_auralguard_result": final_report,
            "quality_report": quality,
        }

        report_path = Path("results") / "demo_clear_last_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

        return (
            final_decision_banner(final_report, args.final_name),
            html_card(args.final_name, final_report, "Main forensic decision", is_final=True),
            html_card(args.baseline_name, baseline_report, "Older model for comparison", is_final=False),
            comparison_panel(baseline_report, final_report),
            final_report.get("beginner_explanation", ""),
            final_report.get("suspicious_segments", []),
            final_report.get("evidence", {}),
            quality_panel(quality),
            str(report_path),
        )

    css = """
    :root {
        --bg-1: #07111f;
        --bg-2: #0b1b32;
        --panel: rgba(15, 28, 50, 0.86);
        --panel-2: rgba(20, 37, 65, 0.94);
        --border: rgba(120, 180, 255, 0.24);
        --text: #f2f7ff;
        --muted: #adc0dc;
        --blue: #4da3ff;
        --cyan: #28e0e8;
        --green: #39d98a;
        --yellow: #ffd166;
        --red: #ff5c7a;
    }

    body, .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(40, 224, 232, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(77, 163, 255, 0.18), transparent 35%),
            linear-gradient(135deg, var(--bg-1), var(--bg-2)) !important;
        color: var(--text) !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }

    .gradio-container {
        max-width: 1260px !important;
    }

    .main-header {
        border: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(20, 44, 77, 0.92), rgba(8, 18, 33, 0.96));
        border-radius: 24px;
        padding: 28px 30px;
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.35);
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
        color: var(--text);
    }

    .subtitle {
        font-size: 16px;
        color: var(--muted);
        margin-top: 12px;
        max-width: 850px;
    }

    .feature-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .feature-chip {
        border: 1px solid rgba(77, 163, 255, 0.24);
        background: rgba(77, 163, 255, 0.10);
        color: #d7e9ff;
        border-radius: 999px;
        padding: 7px 11px;
        font-size: 13px;
        font-weight: 800;
    }

    .final-banner {
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.25);
        background: linear-gradient(135deg, rgba(20,37,65,0.96), rgba(7,17,31,0.96));
    }

    .final-banner.real { border-color: rgba(57, 217, 138, 0.55); }
    .final-banner.fake { border-color: rgba(255, 92, 122, 0.58); }
    .final-banner.review { border-color: rgba(255, 209, 102, 0.55); }

    .final-label {
        color: var(--cyan);
        font-size: 12px;
        font-weight: 950;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .final-decision {
        font-size: 44px;
        font-weight: 950;
        margin-top: 6px;
        letter-spacing: -0.03em;
    }

    .final-banner.real .final-decision { color: var(--green); }
    .final-banner.fake .final-decision { color: var(--red); }
    .final-banner.review .final-decision { color: var(--yellow); }

    .final-subtitle {
        color: var(--muted);
        margin-top: 8px;
        font-size: 15px;
    }

    .final-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 18px;
    }

    .final-grid div {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 14px;
    }

    .final-grid span {
        display: block;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 850;
    }

    .final-grid b {
        display: block;
        margin-top: 6px;
        font-size: 24px;
        color: #fff;
    }

    .panel {
        border: 1px solid var(--border) !important;
        background: var(--panel) !important;
        border-radius: 20px !important;
        box-shadow: 0 18px 45px rgba(0,0,0,0.22) !important;
        padding: 14px !important;
    }

    .result-card, .comparison-card, .quality-card {
        border: 1px solid var(--border);
        background: var(--panel-2);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.22);
        color: var(--text);
    }

    .final-card { border-color: rgba(40, 224, 232, 0.45); }
    .baseline-card { opacity: 0.88; border-style: dashed; }

    .result-card.real { border-color: rgba(57, 217, 138, 0.45); }
    .result-card.fake { border-color: rgba(255, 92, 122, 0.52); }
    .result-card.review { border-color: rgba(255, 209, 102, 0.50); }

    .role-tag {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 11px;
        font-weight: 950;
        letter-spacing: 0.09em;
        margin-bottom: 8px;
    }

    .final-tag { background: rgba(40, 224, 232, 0.15); color: var(--cyan); }
    .baseline-tag { background: rgba(255, 255, 255, 0.09); color: var(--muted); }

    .card-topline {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 8px;
    }

    .model-name {
        font-size: 18px;
        font-weight: 950;
    }

    .card-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 16px;
    }

    .status-pill {
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 950;
        letter-spacing: 0.08em;
        white-space: nowrap;
    }

    .status-pill.real { background: rgba(57, 217, 138, 0.14); color: var(--green); }
    .status-pill.fake { background: rgba(255, 92, 122, 0.14); color: var(--red); }
    .status-pill.review { background: rgba(255, 209, 102, 0.16); color: var(--yellow); }

    .probability-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 14px;
    }

    .metric-label, .small-label {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 850;
    }

    .probability-value {
        font-size: 34px;
        font-weight: 950;
        color: #ffffff;
    }

    .attack-value {
        font-size: 20px;
        font-weight: 900;
        color: #dbeafe;
        margin-top: 6px;
    }

    .score-track {
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        overflow: hidden;
        margin: 12px 0 16px;
    }

    .score-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease;
    }

    .score-fill.real { background: linear-gradient(90deg, var(--green), var(--cyan)); }
    .score-fill.fake { background: linear-gradient(90deg, #ff8a9f, var(--red)); }
    .score-fill.review { background: linear-gradient(90deg, var(--yellow), #ff9f1c); }

    .card-footer {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: var(--muted);
        font-size: 13px;
    }

    .comparison-card h3 {
        margin: 8px 0 6px;
        font-size: 24px;
    }

    .comparison-card p {
        color: var(--muted);
    }

    .comparison-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 10px;
        background: rgba(77, 163, 255, 0.14);
        color: #b9dcff;
        font-weight: 950;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .comparison-grid, .quality-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 14px;
    }

    .comparison-grid div, .quality-metric {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 12px;
    }

    .comparison-grid b {
        display: block;
        margin-top: 5px;
        font-size: 20px;
    }

    .quality-card h3 {
        margin-top: 0;
    }

    .quality-metric span {
        display: block;
        color: var(--muted);
        font-size: 12px;
    }

    .quality-metric b {
        display: block;
        margin-top: 4px;
        font-size: 18px;
    }

    .warning-box {
        border: 1px solid rgba(255, 209, 102, 0.45);
        background: rgba(255, 209, 102, 0.10);
        border-radius: 14px;
        padding: 12px;
        color: #ffe8a3;
    }

    .ok-box {
        border: 1px solid rgba(57, 217, 138, 0.35);
        background: rgba(57, 217, 138, 0.10);
        border-radius: 14px;
        padding: 12px;
        color: #b7f7d4;
    }

    .empty-state {
        border: 1px dashed rgba(159, 180, 208, 0.40);
        border-radius: 18px;
        padding: 18px;
        color: var(--muted);
        background: rgba(255,255,255,0.035);
    }

    textarea, input, .gradio-dropdown, .gradio-textbox {
        background: rgba(255,255,255,0.05) !important;
        color: var(--text) !important;
        border-color: rgba(120, 180, 255, 0.22) !important;
    }

    .prose, .markdown, label, .wrap, .json-holder {
        color: var(--text) !important;
    }

    button.primary {
        background: linear-gradient(90deg, var(--blue), var(--cyan)) !important;
        color: #03101f !important;
        border: none !important;
        font-weight: 950 !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 35px rgba(40, 224, 232, 0.25) !important;
    }
    """

    with gr.Blocks(css=css, title="AuralGuard-AASIST++ Dashboard", theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div class="main-header">
                <div class="eyebrow">Forensic Audio Intelligence</div>
                <h1 class="main-title">AuralGuard-AASIST++</h1>
                <div class="subtitle">
                    Robust and explainable audio deepfake detection for synthetic speech,
                    accented English, and interview-style recordings.
                </div>
                <div class="feature-row">
                    <span class="feature-chip">Final Result First</span>
                    <span class="feature-chip">Baseline Comparison</span>
                    <span class="feature-chip">Evidence Packet</span>
                    <span class="feature-chip">Audio Quality Diagnostics</span>
                    <span class="feature-chip">Suspicious Timestamps</span>
                </div>
            </div>
            """
        )

        final_banner = gr.HTML()

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown("### Upload Audio")
                audio_input = gr.Audio(type="filepath", label="Audio file")
                analyze_btn = gr.Button("Run Forensic Analysis", variant="primary")
                gr.Markdown(
                    """
                    The **Final AuralGuard Result** is the main result.  
                    The baseline is shown only for comparison.
                    """
                )

            with gr.Column(scale=2):
                final_card = gr.HTML()
                baseline_card = gr.HTML()

        with gr.Tab("Model Comparison"):
            comparison = gr.HTML()

        with gr.Tab("Beginner Explanation"):
            beginner_explanation = gr.Textbox(label="Simple explanation", lines=8)

        with gr.Tab("Suspicious Timestamp Evidence"):
            suspicious_segments = gr.JSON(label="Suspicious timestamp regions")

        with gr.Tab("Evidence Packet"):
            evidence_packet = gr.JSON(label="Final model evidence packet")

        with gr.Tab("Audio Quality Diagnostics"):
            quality_report = gr.HTML()

        with gr.Tab("Export"):
            export_path = gr.Textbox(label="Saved JSON report path")

        analyze_btn.click(
            fn=analyze,
            inputs=audio_input,
            outputs=[
                final_banner,
                final_card,
                baseline_card,
                comparison,
                beginner_explanation,
                suspicious_segments,
                evidence_packet,
                quality_report,
                export_path,
            ],
        )

        gr.HTML(
            """
            <div style="margin-top: 18px; color: #adc0dc; font-size: 13px; text-align: center;">
                This dashboard provides forensic decision support, not legal proof.
                Uncertain, noisy, or out-of-domain audio should be reviewed by a human.
            </div>
            """
        )

    demo.launch(share=args.share, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
