from __future__ import annotations

import argparse
import gradio as gr
import torch

from .infer import load_model, run_auralguard
from .explain import beginner_friendly_explanation


def parse_args():
    p = argparse.ArgumentParser(description="Gradio demo for AuralGuard-AASIST++")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--aasist-root", default="external/aasist")
    p.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration-sec", type=float, default=4.0)
    p.add_argument("--feature-dim", type=int, default=160)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def decision_badge(decision, fake_prob):
    if decision == "likely real":
        return f"✅ Likely Real\n\nFake probability: {fake_prob:.4f}"
    if decision == "likely fake":
        return f"🚨 Likely Fake\n\nFake probability: {fake_prob:.4f}"
    return f"⚠️ Suspicious / Human Review\n\nFake probability: {fake_prob:.4f}"


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Using device: {device}")
    model = load_model(args, device)

    def analyze_audio(audio_path):
        if audio_path is None:
            return (
                "Please upload an audio file.",
                0.0,
                "No audio uploaded",
                "No attack type yet.",
                [],
                "No technical explanation yet.",
                "Upload an audio file to get a beginner-friendly explanation.",
                {},
            )

        report = run_auralguard(
            audio_path,
            model,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            device=device,
        )

        decision = report.get("decision", "unknown")
        fake_probability = float(report.get("fake_probability", 0.0))
        attack_type = report.get("attack_type", "unknown")
        suspicious_segments = report.get("suspicious_segments", [])

        beginner_explanation = beginner_friendly_explanation(
            fake_prob=fake_probability,
            attack_type=attack_type,
            suspicious_segments=suspicious_segments,
        )

        technical_explanation = "\n".join(report.get("explanation", []))

        return (
            decision_badge(decision, fake_probability),
            fake_probability,
            decision,
            attack_type,
            suspicious_segments,
            technical_explanation,
            beginner_explanation,
            report.get("evidence", {}),
        )

    css = """
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #666;
        margin-bottom: 25px;
    }
    .result-box {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }
    """

    with gr.Blocks(css=css, title="AuralGuard-AASIST++") as demo:
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
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload Audio File"
                )

                analyze_button = gr.Button(
                    "Analyze Audio",
                    variant="primary"
                )

                gr.Markdown(
                    """
                    ### What this demo shows
                    - Fake / real decision  
                    - Fake probability  
                    - Attack-type clue  
                    - Suspicious timestamp regions  
                    - Beginner-friendly explanation  
                    - Evidence packet for technical review  
                    """
                )

            with gr.Column(scale=1):
                result_summary = gr.Textbox(
                    label="Main Result",
                    lines=4,
                    elem_classes=["result-box"]
                )

                fake_probability = gr.Number(
                    label="Fake Probability"
                )

                decision = gr.Textbox(
                    label="Decision"
                )

                attack_type = gr.Textbox(
                    label="Attack Type"
                )

        with gr.Tab("Beginner Explanation"):
            beginner_explanation = gr.Textbox(
                label="Simple Explanation",
                lines=7
            )

        with gr.Tab("Technical Explanation"):
            technical_explanation = gr.Textbox(
                label="Technical Explanation",
                lines=7
            )

        with gr.Tab("Suspicious Segments"):
            suspicious_segments = gr.JSON(
                label="Suspicious Timestamp Regions"
            )

        with gr.Tab("Evidence Packet"):
            evidence_packet = gr.JSON(
                label="Evidence Packet"
            )

        analyze_button.click(
            fn=analyze_audio,
            inputs=audio_input,
            outputs=[
                result_summary,
                fake_probability,
                decision,
                attack_type,
                suspicious_segments,
                technical_explanation,
                beginner_explanation,
                evidence_packet,
            ],
        )

        gr.Markdown(
            """
            ---
            **Note:** A high fake score should be treated as forensic evidence, not final proof.  
            For real-world accented, noisy, or interview-style audio, human review is recommended when the model is uncertain.
            """
        )

    demo.launch(
        share=True,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()