
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch

from .infer import load_model, run_auralguard
from .audio_quality import audio_quality_report
from .decision_gate import make_safe_report
from .explain_plus import beginner_friendly_explanation_plus


def parse_args():
    p = argparse.ArgumentParser(description="Improved AuralGuard-AASIST++ Gradio demo with baseline-vs-final comparison.")
    p.add_argument("--baseline-checkpoint", default="", help="Optional baseline checkpoint, e.g. ASVspoof-only best.pt")
    p.add_argument("--final-checkpoint", required=True, help="Final/best AuralGuard checkpoint")
    p.add_argument("--baseline-name", default="ASVspoof baseline")
    p.add_argument("--final-name", default="AuralGuard final balanced")
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


def emoji_for_decision(decision: str) -> str:
    d = (decision or "").lower()
    if "real" in d:
        return "✅"
    if "fake" in d:
        return "🚨"
    return "⚠️"


def result_card(model_name: str, report: dict) -> str:
    decision = report.get("decision", "unknown")
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = report.get("attack_type", "unknown")
    review = report.get("review_required", False)
    emoji = emoji_for_decision(decision)

    return (
        f"{emoji} **{model_name}**\\n\\n"
        f"**Decision:** {decision}\\n\\n"
        f"**Fake probability:** {fake_prob:.4f}\\n\\n"
        f"**Attack-type clue:** {attack_type}\\n\\n"
        f"**Human review required:** {'Yes' if review else 'No'}"
    )


def make_report_for_model(audio_path, model, model_name, args, device, quality):
    raw = run_auralguard(
        audio_path,
        model,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        device=device,
    )
    safe = make_safe_report(raw, quality_warnings=quality["warnings"], ood_warning=False)
    gate = safe.get("decision_gate", {})
    beginner = beginner_friendly_explanation_plus(
        fake_prob=float(safe.get("fake_probability", 0.0)),
        attack_type=safe.get("attack_type", "unknown"),
        suspicious_segments=safe.get("suspicious_segments", []),
        safe_decision=safe.get("decision", "unknown"),
        warnings=gate.get("warnings", []),
    )

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
            empty = "Upload an audio file first."
            return empty, empty, "", "", {}, {}, "", None

        quality = audio_quality_report(audio_path, target_sr=args.sample_rate)

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

        base_prob = float(baseline_report.get("fake_probability", 0.0))
        final_prob = float(final_report.get("fake_probability", 0.0))
        improvement = base_prob - final_prob

        comparison_text = (
            f"### Comparison summary\\n"
            f"- Baseline fake probability: **{base_prob:.4f}**\\n"
            f"- Final model fake probability: **{final_prob:.4f}**\\n"
            f"- Difference baseline - final: **{improvement:.4f}**\\n\\n"
        )

        if improvement > 0.20:
            comparison_text += (
                "The final model is much less likely to call this clip fake. "
                "This supports the false-alarm reduction goal."
            )
        elif improvement < -0.20:
            comparison_text += (
                "The final model is more suspicious of this clip than the baseline. "
                "This should be reviewed carefully."
            )
        else:
            comparison_text += "Both models give broadly similar confidence."

        quality_text = "### Audio quality report\\n"
        metrics = quality.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, float):
                quality_text += f"- {k}: {v:.4f}\\n"
            else:
                quality_text += f"- {k}: {v}\\n"

        warnings = quality.get("warnings", [])
        if warnings:
            quality_text += "\\n### Reliability warnings\\n" + "\\n".join([f"- {w}" for w in warnings])
        else:
            quality_text += "\\nNo major audio-quality warnings detected."

        combined = {
            "baseline": baseline_report,
            "final": final_report,
            "comparison": {
                "baseline_fake_probability": base_prob,
                "final_fake_probability": final_prob,
                "difference_baseline_minus_final": improvement,
            },
            "quality_report": quality,
        }

        report_path = Path("results") / "demo_last_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

        return (
            result_card(args.baseline_name, baseline_report),
            result_card(args.final_name, final_report),
            comparison_text,
            final_report.get("beginner_explanation", ""),
            final_report.get("suspicious_segments", []),
            final_report.get("evidence", {}),
            quality_text,
            str(report_path),
        )

    css = """
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 22px;
    }
    .note-box {
        border-radius: 12px;
        padding: 12px;
        background: #f7f7f7;
    }
    """

    with gr.Blocks(css=css, title="AuralGuard-AASIST++ Demo") as demo:
        gr.HTML(
            """
            <div class="main-title">AuralGuard-AASIST++</div>
            <div class="subtitle">
            Robust and Explainable Audio Deepfake Detection for Synthetic Speech and Real-World Accented Speech
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(type="filepath", label="Upload audio")
                analyze_btn = gr.Button("Analyze Audio", variant="primary")
                gr.Markdown(
                    """
                    ### What this improved demo shows
                    - Baseline vs final model comparison
                    - Traffic-light fake/real/human-review decision
                    - Beginner-friendly explanation
                    - Suspicious timestamp regions
                    - Audio-quality warnings
                    - Exported JSON evidence report
                    """
                )

            with gr.Column(scale=1):
                baseline_card = gr.Markdown(label="Baseline result")
                final_card = gr.Markdown(label="Final model result")

        with gr.Tab("Comparison Summary"):
            comparison = gr.Markdown()

        with gr.Tab("Beginner Explanation"):
            beginner_explanation = gr.Textbox(label="Simple explanation", lines=8)

        with gr.Tab("Suspicious Segments"):
            suspicious_segments = gr.JSON(label="Suspicious timestamp regions")

        with gr.Tab("Evidence Packet"):
            evidence_packet = gr.JSON(label="Final model evidence packet")

        with gr.Tab("Audio Quality"):
            quality_report = gr.Markdown()

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

        gr.Markdown(
            """
            ---
            **Important:** This system gives forensic evidence, not legal proof.  
            For real-world accented, noisy, old, or interview-style audio, human review is recommended when the model is uncertain.
            """
        )

    demo.launch(share=args.share, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
