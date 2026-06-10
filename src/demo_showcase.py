
from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser(description="Visual showcase demo for AuralGuard-AASIST++")
    p.add_argument("--baseline-checkpoint", default="", help="Optional ASVspoof-only baseline checkpoint")
    p.add_argument("--final-checkpoint", required=True, help="Final AuralGuard checkpoint")
    p.add_argument("--baseline-name", default="ASVspoof Baseline")
    p.add_argument("--final-name", default="AuralGuard Final")
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


def safe_quality_report(audio_path: str, sample_rate: int):
    if audio_quality_report is None:
        return {"warnings": [], "metrics": {}}
    try:
        return audio_quality_report(audio_path, target_sr=sample_rate)
    except Exception as e:
        return {
            "warnings": [f"Could not compute audio quality metrics: {e}"],
            "metrics": {},
        }


def safe_gate(report: dict, quality: dict):
    if make_safe_report is None:
        return report
    try:
        return make_safe_report(report, quality_warnings=quality.get("warnings", []), ood_warning=False)
    except Exception:
        return report


def safe_beginner_explanation(report: dict, quality: dict):
    if beginner_friendly_explanation_plus is not None:
        try:
            gate = report.get("decision_gate", {})
            return beginner_friendly_explanation_plus(
                fake_prob=float(report.get("fake_probability", 0.0)),
                attack_type=report.get("attack_type", "unknown"),
                suspicious_segments=report.get("suspicious_segments", []),
                safe_decision=report.get("decision", "unknown"),
                warnings=gate.get("warnings", []) + quality.get("warnings", []),
            )
        except Exception:
            pass

    fake_prob = float(report.get("fake_probability", 0.0))
    decision = report.get("decision", "unknown")
    attack_type = report.get("attack_type", "unknown")

    if "real" in decision:
        return (
            "The model thinks this audio is likely real because the fake score is low and the sound patterns "
            "look closer to real human speech examples."
        )
    if "fake" in decision:
        return (
            f"The model thinks this audio may be fake because the fake score is high and the attack-type head "
            f"suggests {attack_type} style evidence."
        )
    return (
        "The model is not fully sure. The audio should be reviewed by a human rather than automatically labelled fake or real."
    )


def decision_color(decision: str):
    d = (decision or "").lower()
    if "real" in d:
        return "#16a34a", "✅", "Likely Real"
    if "fake" in d:
        return "#dc2626", "🚨", "Likely Fake"
    return "#f59e0b", "⚠️", "Human Review"


def probability_bar(prob: float):
    prob = max(0.0, min(1.0, float(prob)))
    if prob < 0.35:
        color = "#16a34a"
    elif prob < 0.85:
        color = "#f59e0b"
    else:
        color = "#dc2626"

    return f"""
    <div class="prob-wrap">
        <div class="prob-label">
            <span>Fake probability</span>
            <strong>{prob:.4f}</strong>
        </div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{prob*100:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-scale">
            <span>0.00 Real</span><span>0.35</span><span>0.85</span><span>1.00 Fake</span>
        </div>
    </div>
    """


def model_card(name: str, report: dict):
    decision = report.get("decision", "unknown")
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = report.get("attack_type", "unknown")
    review = report.get("review_required", False)
    color, icon, label = decision_color(decision)

    return f"""
    <div class="model-card">
        <div class="card-topline">
            <div>
                <div class="model-name">{name}</div>
                <div class="model-subtitle">AASIST-based detector</div>
            </div>
            <div class="badge" style="background:{color};">{icon} {label}</div>
        </div>

        {probability_bar(fake_prob)}

        <div class="mini-grid">
            <div class="mini-stat">
                <div class="mini-label">Decision</div>
                <div class="mini-value">{decision}</div>
            </div>
            <div class="mini-stat">
                <div class="mini-label">Attack clue</div>
                <div class="mini-value">{attack_type}</div>
            </div>
            <div class="mini-stat">
                <div class="mini-label">Human review</div>
                <div class="mini-value">{'Yes' if review else 'No'}</div>
            </div>
        </div>
    </div>
    """


def comparison_banner(baseline_report: dict, final_report: dict):
    b = float(baseline_report.get("fake_probability", 0.0))
    f = float(final_report.get("fake_probability", 0.0))
    diff = b - f

    if diff > 0.20:
        headline = "False-alarm reduction visible"
        text = (
            f"The final model lowered the fake probability by {diff:.4f}. "
            "This supports the project goal of reducing false alarms on real-world speech."
        )
        color = "#0f766e"
        icon = "📉"
    elif diff < -0.20:
        headline = "Final model is more suspicious"
        text = (
            f"The final model increased the fake probability by {abs(diff):.4f}. "
            "This clip should be reviewed carefully."
        )
        color = "#b45309"
        icon = "🔎"
    else:
        headline = "Models broadly agree"
        text = (
            f"The fake probability changed by {diff:.4f}. "
            "Both models give similar confidence on this clip."
        )
        color = "#2563eb"
        icon = "📊"

    return f"""
    <div class="comparison-banner" style="border-left-color:{color};">
        <div class="comparison-icon">{icon}</div>
        <div>
            <div class="comparison-title">{headline}</div>
            <div class="comparison-text">{text}</div>
        </div>
    </div>
    """


def quality_html(quality: dict):
    metrics = quality.get("metrics", {}) or {}
    warnings = quality.get("warnings", []) or []

    cards = ""
    for k, v in metrics.items():
        if isinstance(v, float):
            show = f"{v:.4f}"
        else:
            show = str(v)
        cards += f"""
        <div class="quality-card">
            <div class="quality-label">{k.replace('_', ' ').title()}</div>
            <div class="quality-value">{show}</div>
        </div>
        """

    if not cards:
        cards = "<p>No audio quality metrics available.</p>"

    if warnings:
        warning_list = "".join([f"<li>{w}</li>" for w in warnings])
        warning_box = f"""
        <div class="warning-box">
            <strong>Reliability warnings</strong>
            <ul>{warning_list}</ul>
        </div>
        """
    else:
        warning_box = """
        <div class="ok-box">
            <strong>No major audio-quality warnings detected.</strong>
        </div>
        """

    return f"""
    <div class="quality-grid">{cards}</div>
    {warning_box}
    """


def segment_timeline(suspicious_segments):
    suspicious_segments = suspicious_segments or []
    if not suspicious_segments:
        return """
        <div class="empty-state">
            No suspicious timestamp region was detected for this clip.
        </div>
        """

    rows = ""
    for i, s in enumerate(suspicious_segments[:10], start=1):
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        score = float(s.get("fake_score", 0.0))
        color = "#dc2626" if score >= 0.85 else "#f59e0b"
        rows += f"""
        <div class="segment-row">
            <div class="segment-id">#{i}</div>
            <div class="segment-time">{start:.2f}s → {end:.2f}s</div>
            <div class="segment-score" style="color:{color};">{score:.3f}</div>
        </div>
        """

    return f"""
    <div class="segment-box">
        <div class="segment-header">
            <span>Suspicious region</span>
            <span>Fake score</span>
        </div>
        {rows}
    </div>
    """


def make_report_for_model(audio_path, model, model_name, args, device, quality):
    raw = run_auralguard(
        audio_path,
        model,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        device=device,
    )
    safe = safe_gate(raw, quality)
    beginner = safe_beginner_explanation(safe, quality)

    safe["model_name"] = model_name
    safe["beginner_explanation"] = beginner
    safe["quality_report"] = quality
    return safe


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
            empty_card = """
            <div class="empty-state big">
                Upload an audio clip to begin analysis.
            </div>
            """
            return empty_card, empty_card, "", "", "", {}, "", None

        quality = safe_quality_report(audio_path, args.sample_rate)

        final_report = make_report_for_model(
            audio_path, final_model, args.final_name, args, device, quality
        )

        if baseline_model is not None:
            baseline_report = make_report_for_model(
                audio_path, baseline_model, args.baseline_name, args, device, quality
            )
        else:
            baseline_report = {
                "model_name": args.baseline_name,
                "decision": "not loaded",
                "fake_probability": 0.0,
                "attack_type": "not loaded",
                "review_required": False,
                "beginner_explanation": "No baseline checkpoint was provided.",
                "suspicious_segments": [],
                "evidence": {},
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

        report_path = Path("results") / "demo_showcase_last_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

        return (
            model_card(args.baseline_name, baseline_report),
            model_card(args.final_name, final_report),
            comparison_banner(baseline_report, final_report),
            final_report.get("beginner_explanation", ""),
            segment_timeline(final_report.get("suspicious_segments", [])),
            final_report.get("evidence", {}),
            quality_html(quality),
            str(report_path),
        )

    css = """
    body {
        background: radial-gradient(circle at top left, #e0f2fe 0, transparent 32%),
                    radial-gradient(circle at top right, #fee2e2 0, transparent 28%),
                    linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%) !important;
    }

    .gradio-container {
        max-width: 1180px !important;
        margin: auto !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }

    .hero {
        padding: 30px 28px;
        border-radius: 28px;
        color: white;
        background:
            linear-gradient(135deg, rgba(15,23,42,.94), rgba(30,64,175,.86)),
            radial-gradient(circle at 20% 20%, rgba(56,189,248,.32), transparent 30%);
        box-shadow: 0 24px 60px rgba(15,23,42,.25);
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,.18);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: -1.2px;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        line-height: 1.55;
        opacity: .92;
        max-width: 880px;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }

    .pill {
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.13);
        border: 1px solid rgba(255,255,255,.22);
        font-size: 13px;
        font-weight: 700;
    }

    .upload-card {
        padding: 18px;
        border-radius: 22px;
        background: rgba(255,255,255,.82);
        border: 1px solid rgba(148,163,184,.25);
        box-shadow: 0 16px 40px rgba(15,23,42,.08);
    }

    .model-card {
        padding: 20px;
        border-radius: 24px;
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(148,163,184,.25);
        box-shadow: 0 18px 50px rgba(15,23,42,.10);
        margin-bottom: 14px;
    }

    .card-topline {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 16px;
    }

    .model-name {
        font-size: 20px;
        font-weight: 850;
        color: #0f172a;
    }

    .model-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-top: 2px;
    }

    .badge {
        color: white;
        padding: 8px 12px;
        border-radius: 999px;
        font-weight: 850;
        font-size: 13px;
        box-shadow: 0 8px 18px rgba(15,23,42,.15);
        white-space: nowrap;
    }

    .prob-wrap {
        margin: 16px 0;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        color: #334155;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .prob-track {
        height: 13px;
        border-radius: 999px;
        background: #e2e8f0;
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        border-radius: 999px;
        transition: width .45s ease;
    }

    .prob-scale {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
    }

    .mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 14px;
    }

    .mini-stat {
        padding: 12px;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .mini-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: .04em;
        font-weight: 800;
    }

    .mini-value {
        font-size: 13px;
        color: #0f172a;
        margin-top: 5px;
        font-weight: 800;
        word-break: break-word;
    }

    .comparison-banner {
        display: flex;
        gap: 16px;
        align-items: flex-start;
        padding: 18px;
        border-radius: 22px;
        background: rgba(255,255,255,.9);
        border: 1px solid rgba(148,163,184,.25);
        border-left: 8px solid;
        box-shadow: 0 14px 36px rgba(15,23,42,.08);
    }

    .comparison-icon {
        font-size: 32px;
    }

    .comparison-title {
        font-size: 20px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .comparison-text {
        color: #475569;
        line-height: 1.55;
    }

    .quality-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 14px;
    }

    .quality-card {
        padding: 14px;
        border-radius: 18px;
        background: white;
        border: 1px solid #e2e8f0;
    }

    .quality-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 850;
    }

    .quality-value {
        font-size: 20px;
        font-weight: 900;
        color: #0f172a;
        margin-top: 4px;
    }

    .warning-box {
        padding: 14px;
        border-radius: 18px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
    }

    .ok-box {
        padding: 14px;
        border-radius: 18px;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
    }

    .segment-box {
        padding: 16px;
        border-radius: 20px;
        background: white;
        border: 1px solid #e2e8f0;
    }

    .segment-header, .segment-row {
        display: grid;
        grid-template-columns: 80px 1fr 120px;
        gap: 10px;
        align-items: center;
    }

    .segment-header {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .segment-row {
        padding: 10px 0;
        border-top: 1px solid #e2e8f0;
    }

    .segment-id {
        font-weight: 900;
        color: #334155;
    }

    .segment-time {
        color: #0f172a;
        font-weight: 750;
    }

    .segment-score {
        font-weight: 900;
    }

    .empty-state {
        padding: 18px;
        border-radius: 18px;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        color: #64748b;
        text-align: center;
        font-weight: 700;
    }

    .empty-state.big {
        padding: 45px 20px;
        font-size: 18px;
    }

    textarea, input {
        border-radius: 14px !important;
    }

    .footer-note {
        padding: 16px 18px;
        border-radius: 20px;
        background: rgba(15,23,42,.90);
        color: white;
        line-height: 1.55;
        margin-top: 16px;
    }
    """

    with gr.Blocks(css=css, title="AuralGuard-AASIST++ Showcase") as demo:
        gr.HTML(
            """
            <div class="hero">
                <div class="hero-title">AuralGuard-AASIST++</div>
                <div class="hero-subtitle">
                    A robust and explainable audio deepfake detector designed for synthetic speech,
                    real-world accented speech, interview audio, and human-review forensic workflows.
                </div>
                <div class="hero-pills">
                    <span class="pill">🎙️ Audio Deepfake Detection</span>
                    <span class="pill">🌍 Accent Robustness</span>
                    <span class="pill">⚖️ False-Alarm Reduction</span>
                    <span class="pill">🧠 Beginner Explanation</span>
                    <span class="pill">📍 Suspicious Timestamps</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group(elem_classes=["upload-card"]):
                    audio_input = gr.Audio(type="filepath", label="Upload an audio clip")
                    analyze_btn = gr.Button("Analyze Audio", variant="primary", size="lg")
                    gr.Markdown(
                        """
                        **Best demo test:** upload the same DECTE/FRED/interview clip and compare the baseline model with the final balanced model.
                        """
                    )

            with gr.Column(scale=6):
                baseline_card = gr.HTML()
                final_card = gr.HTML()

        comparison = gr.HTML()

        with gr.Tab("Beginner Explanation"):
            beginner_explanation = gr.Textbox(
                label="Simple explanation for non-technical users",
                lines=8,
                show_copy_button=True,
            )

        with gr.Tab("Suspicious Timestamp Regions"):
            suspicious_segments_html = gr.HTML()

        with gr.Tab("Evidence Packet"):
            evidence_packet = gr.JSON(label="Final model evidence packet")

        with gr.Tab("Audio Quality"):
            quality_report_html = gr.HTML()

        with gr.Tab("Export"):
            export_path = gr.Textbox(label="Saved JSON report path", show_copy_button=True)
            gr.Markdown(
                """
                The JSON report can be used as evidence for screenshots, presentations, or your paper appendix.
                """
            )

        analyze_btn.click(
            fn=analyze,
            inputs=audio_input,
            outputs=[
                baseline_card,
                final_card,
                comparison,
                beginner_explanation,
                suspicious_segments_html,
                evidence_packet,
                quality_report_html,
                export_path,
            ],
        )

        gr.HTML(
            """
            <div class="footer-note">
                <strong>Important:</strong> This system provides forensic evidence, not legal proof.
                When the decision is uncertain, or when the audio is noisy, old, accented, compressed, or interview-style,
                human review is recommended.
            </div>
            """
        )

    demo.launch(share=args.share, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
