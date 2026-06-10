
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
    p = argparse.ArgumentParser(description="Polished AuralGuard-AASIST++ forensic dashboard demo")
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
        return (
            "The model gives this audio a low fake score, so it appears closer to real human speech "
            "than to generated or manipulated speech."
        )
    if "fake" in decision.lower():
        return (
            "The model gives this audio a high fake score and found patterns similar to generated "
            "or manipulated speech examples."
        )
    return (
        "The model is not fully confident. The safest interpretation is to treat this clip as requiring human review."
    )


def status_class(decision: str) -> tuple[str, str]:
    d = (decision or "").lower()
    if "real" in d:
        return "real", "LIKELY REAL"
    if "fake" in d:
        return "fake", "LIKELY FAKE"
    return "review", "HUMAN REVIEW"


def html_card(model_name: str, report: dict, subtitle: str) -> str:
    decision = str(report.get("decision", "unknown"))
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = str(report.get("attack_type", "unknown"))
    review = bool(report.get("review_required", False))
    cls, label = status_class(decision)

    bar_width = max(0.0, min(100.0, fake_prob * 100.0))
    escaped_model = html.escape(model_name)
    escaped_attack = html.escape(attack_type)
    escaped_subtitle = html.escape(subtitle)

    return f"""
    <div class="result-card {cls}">
        <div class="card-topline">
            <span class="model-name">{escaped_model}</span>
            <span class="status-pill {cls}">{label}</span>
        </div>
        <div class="card-subtitle">{escaped_subtitle}</div>

        <div class="probability-row">
            <div>
                <div class="metric-label">Fake probability</div>
                <div class="probability-value">{fake_prob:.4f}</div>
            </div>
            <div>
                <div class="metric-label">Attack-type clue</div>
                <div class="attack-value">{escaped_attack}</div>
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

    if diff > 0.20:
        headline = "Final model is less suspicious of this clip"
        detail = "This suggests the final model may be reducing unnecessary false alarms."
        badge = "Reduced fake score"
        cls = "positive"
    elif diff < -0.20:
        headline = "Final model is more suspicious of this clip"
        detail = "This suggests the final model found stronger fake-like evidence than the baseline."
        badge = "Higher fake score"
        cls = "caution"
    else:
        headline = "Both models give similar confidence"
        detail = "The fake probabilities are close, so the models mostly agree."
        badge = "Similar confidence"
        cls = "neutral"

    return f"""
    <div class="comparison-card {cls}">
        <div class="comparison-badge">{badge}</div>
        <h3>{headline}</h3>
        <p>{detail}</p>
        <div class="comparison-grid">
            <div>
                <span class="small-label">Baseline fake probability</span>
                <b>{base_prob:.4f}</b>
            </div>
            <div>
                <span class="small-label">Final fake probability</span>
                <b>{final_prob:.4f}</b>
            </div>
            <div>
                <span class="small-label">Difference</span>
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
        if isinstance(value, float):
            val = f"{value:.4f}"
        else:
            val = str(value)
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
            return (
                "<div class='empty-state'>Upload an audio file to begin analysis.</div>",
                "<div class='empty-state'>No final model result yet.</div>",
                "<div class='empty-state'>No comparison yet.</div>",
                "Upload an audio file to generate an explanation.",
                [],
                {},
                "<div class='empty-state'>No audio-quality report yet.</div>",
                "",
            )

        quality = safe_quality_report(audio_path, args.sample_rate)

        raw_final = run_auralguard(
            audio_path,
            final_model,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            device=device,
        )
        final_report = safe_make_report(raw_final, quality)
        final_report["model_name"] = args.final_name
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
            baseline_report["model_name"] = args.baseline_name
            baseline_report["beginner_explanation"] = simple_explanation(baseline_report)
            baseline_report["quality_report"] = quality
        else:
            baseline_report = {
                "decision": "not loaded",
                "fake_probability": 0.0,
                "attack_type": "not loaded",
                "review_required": False,
                "suspicious_segments": [],
                "evidence": {},
                "beginner_explanation": "No baseline checkpoint was provided.",
                "quality_report": quality,
            }

        combined = {
            "baseline": baseline_report,
            "final": final_report,
            "quality_report": quality,
            "comparison": {
                "baseline_fake_probability": float(baseline_report.get("fake_probability", 0.0)),
                "final_fake_probability": float(final_report.get("fake_probability", 0.0)),
                "difference_baseline_minus_final": float(baseline_report.get("fake_probability", 0.0))
                - float(final_report.get("fake_probability", 0.0)),
            },
        }

        report_path = Path("results") / "demo_technical_last_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

        return (
            html_card(args.baseline_name, baseline_report, "Reference model output"),
            html_card(args.final_name, final_report, "Robust final model output"),
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
        --panel: rgba(15, 28, 50, 0.82);
        --panel-2: rgba(20, 37, 65, 0.92);
        --border: rgba(120, 180, 255, 0.22);
        --text: #eaf2ff;
        --muted: #9fb4d0;
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
        max-width: 1250px !important;
    }

    .main-header {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(20, 44, 77, 0.88), rgba(8, 18, 33, 0.94));
        border-radius: 24px;
        padding: 28px 30px;
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.35);
        margin-bottom: 18px;
    }

    .main-header:before {
        content: "";
        position: absolute;
        inset: -2px;
        background: linear-gradient(90deg, transparent, rgba(40, 224, 232, 0.18), transparent);
        transform: translateX(-70%);
        animation: sweep 6s infinite;
    }

    @keyframes sweep {
        0% { transform: translateX(-70%); }
        55% { transform: translateX(70%); }
        100% { transform: translateX(70%); }
    }

    .header-content {
        position: relative;
        z-index: 2;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 13px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .main-title {
        font-size: 42px;
        line-height: 1.05;
        margin: 0;
        font-weight: 900;
        color: var(--text);
    }

    .subtitle {
        font-size: 16px;
        color: var(--muted);
        margin-top: 12px;
        max-width: 820px;
    }

    .feature-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .feature-chip {
        border: 1px solid rgba(77, 163, 255, 0.22);
        background: rgba(77, 163, 255, 0.10);
        color: #d7e9ff;
        border-radius: 999px;
        padding: 7px 11px;
        font-size: 13px;
        font-weight: 700;
    }

    .panel {
        border: 1px solid var(--border) !important;
        background: var(--panel) !important;
        border-radius: 20px !important;
        box-shadow: 0 18px 45px rgba(0,0,0,0.22) !important;
    }

    .result-card, .comparison-card, .quality-card {
        border: 1px solid var(--border);
        background: var(--panel-2);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.22);
        color: var(--text);
    }

    .result-card.real { border-color: rgba(57, 217, 138, 0.45); }
    .result-card.fake { border-color: rgba(255, 92, 122, 0.52); }
    .result-card.review { border-color: rgba(255, 209, 102, 0.50); }

    .card-topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 14px;
        margin-bottom: 8px;
    }

    .model-name {
        font-size: 18px;
        font-weight: 900;
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
        font-weight: 900;
        letter-spacing: 0.08em;
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
        font-weight: 800;
    }

    .probability-value {
        font-size: 34px;
        font-weight: 950;
        color: #ffffff;
    }

    .attack-value {
        font-size: 20px;
        font-weight: 850;
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
        font-weight: 900;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .comparison-card.positive { border-color: rgba(57, 217, 138, 0.45); }
    .comparison-card.caution { border-color: rgba(255, 209, 102, 0.5); }

    .comparison-grid {
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
        font-size: 22px;
    }

    .quality-card h3 {
        margin-top: 0;
    }

    .quality-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 14px;
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
        font-weight: 900 !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 35px rgba(40, 224, 232, 0.25) !important;
    }
    """

    with gr.Blocks(css=css, title="AuralGuard-AASIST++ Forensic Dashboard", theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div class="main-header">
                <div class="header-content">
                    <div class="eyebrow">Forensic Audio Intelligence</div>
                    <h1 class="main-title">AuralGuard-AASIST++</h1>
                    <div class="subtitle">
                        Robust and explainable audio deepfake detection for synthetic speech,
                        accented English, and interview-style recordings.
                    </div>
                    <div class="feature-row">
                        <span class="feature-chip">Fake / Real / Human Review</span>
                        <span class="feature-chip">Baseline vs Final Model</span>
                        <span class="feature-chip">Evidence Packet</span>
                        <span class="feature-chip">Audio Quality Diagnostics</span>
                        <span class="feature-chip">Suspicious Timestamps</span>
                    </div>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown("### Upload Audio")
                audio_input = gr.Audio(type="filepath", label="Audio file")
                analyze_btn = gr.Button("Run Forensic Analysis", variant="primary")
                gr.Markdown(
                    """
                    **Output includes:**  
                    fake probability, decision gate, attack-type clue, reliability warnings,
                    timestamp evidence, and a JSON report.
                    """
                )

            with gr.Column(scale=2):
                with gr.Row():
                    baseline_card = gr.HTML()
                    final_card = gr.HTML()

        with gr.Tab("Comparison"):
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
                baseline_card,
                final_card,
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
            <div style="margin-top: 18px; color: #9fb4d0; font-size: 13px; text-align: center;">
                This dashboard provides forensic decision support, not legal proof.
                Uncertain, noisy, or out-of-domain audio should be reviewed by a human.
            </div>
            """
        )

    demo.launch(share=args.share, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
