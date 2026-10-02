from __future__ import annotations

FORCE_READABILITY_STYLE_TAG = """
<style>
.gradio-container, .gradio-container * {
  color: #f8fafc !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container pre,
.gradio-container code {
  color: #f8fafc !important;
  background-color: #0f172a !important;
}
</style>
"""


import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch

from .infer import load_model, run_auralguard

SPECIFIC_READABILITY_FIX_CSS = '\n/* FINAL SPECIFIC READABILITY FIX */\n\n/* Make specific problem areas black/dark with white text */\n#evidence_packet_box,\n#evidence_packet_box *,\n#suspicious_segments_box,\n#suspicious_segments_box *,\n#beginner_explanation_box,\n#beginner_explanation_box *,\n#export_path_box,\n#export_path_box *,\n#voice_pattern_box,\n#voice_pattern_box * {\n  background: #050b14 !important;\n  background-color: #050b14 !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n}\n\n/* Gradio JSON / code internal elements */\n#evidence_packet_box pre,\n#evidence_packet_box code,\n#evidence_packet_box span,\n#evidence_packet_box div,\n#evidence_packet_box textarea,\n#evidence_packet_box input,\n#suspicious_segments_box pre,\n#suspicious_segments_box code,\n#suspicious_segments_box span,\n#suspicious_segments_box div,\n#suspicious_segments_box textarea,\n#suspicious_segments_box input {\n  background: #050b14 !important;\n  background-color: #050b14 !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  caret-color: #ffffff !important;\n  border-color: #60a5fa !important;\n}\n\n/* CodeMirror / JSON viewer internals used by Gradio */\n#evidence_packet_box .cm-editor,\n#evidence_packet_box .cm-scroller,\n#evidence_packet_box .cm-content,\n#evidence_packet_box .cm-line,\n#evidence_packet_box .cm-gutters,\n#evidence_packet_box .json-holder,\n#evidence_packet_box .json-holder *,\n#suspicious_segments_box .cm-editor,\n#suspicious_segments_box .cm-scroller,\n#suspicious_segments_box .cm-content,\n#suspicious_segments_box .cm-line,\n#suspicious_segments_box .cm-gutters,\n#suspicious_segments_box .json-holder,\n#suspicious_segments_box .json-holder * {\n  background: #050b14 !important;\n  background-color: #050b14 !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n}\n\n/* Force labels readable */\n#evidence_packet_box .block-label,\n#suspicious_segments_box .block-label,\n#beginner_explanation_box .block-label,\n#export_path_box .block-label {\n  background: #1e3a8a !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n}\n\n/* Voice-pattern/prosody interpretation */\n.voice-pattern-interpretation,\n.voice-pattern-interpretation *,\n.prosody-interpretation,\n.prosody-interpretation * {\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n}\n\n/* General fallback for textboxes */\n.gradio-container textarea,\n.gradio-container input,\n.gradio-container textarea:disabled,\n.gradio-container input:disabled {\n  background: #050b14 !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n}\n'


FORCE_DARK_READABLE_CSS = '\n/* FORCE DARK READABLE AURALGUARD DEMO */\n\n:root {\n  --ag-page: #061426;\n  --ag-panel: #071629;\n  --ag-border: #60a5fa;\n  --ag-text: #ffffff;\n}\n\n/* Entire app */\n.gradio-container,\n.gradio-container * {\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n}\n\n/* Main background */\n.gradio-container {\n  background: #061426 !important;\n}\n\n/* All component shells */\n.gradio-container .block,\n.gradio-container .form,\n.gradio-container .wrap,\n.gradio-container .panel,\n.gradio-container .container,\n.gradio-container .input-container,\n.gradio-container .output-class,\n.gradio-container .tabs,\n.gradio-container .tabitem {\n  background: #071629 !important;\n  color: #ffffff !important;\n  border-color: #334155 !important;\n}\n\n/* Textboxes and inputs */\n.gradio-container textarea,\n.gradio-container input,\n.gradio-container textarea:disabled,\n.gradio-container input:disabled,\n.gradio-container [data-testid="textbox"] textarea,\n.gradio-container [data-testid="textbox"] input {\n  background-color: #071629 !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n  caret-color: #ffffff !important;\n  border: 1px solid #60a5fa !important;\n  box-shadow: none !important;\n}\n\n/* White inner areas around textboxes */\n.gradio-container [data-testid="textbox"],\n.gradio-container [data-testid="textbox"] > div,\n.gradio-container [data-testid="textbox"] label,\n.gradio-container [data-testid="textbox"] .wrap,\n.gradio-container [data-testid="textbox"] .container,\n.gradio-container [data-testid="textbox"] .input-container {\n  background: #071629 !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n}\n\n/* JSON/code/evidence packet blocks */\n.gradio-container pre,\n.gradio-container code,\n.gradio-container pre *,\n.gradio-container code *,\n.gradio-container .json-holder,\n.gradio-container .json-holder *,\n.gradio-container .cm-editor,\n.gradio-container .cm-editor *,\n.gradio-container .cm-scroller,\n.gradio-container .cm-content,\n.gradio-container .cm-line {\n  background: #071629 !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n}\n\n/* Labels */\n.gradio-container .block-label,\n.gradio-container .block-title,\n.gradio-container .label-wrap,\n.gradio-container .label-wrap *,\n.gradio-container label,\n.gradio-container label * {\n  background: #1e3a8a !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n  border-radius: 8px !important;\n}\n\n/* Tabs */\n.gradio-container .tab-nav button,\n.gradio-container button[role="tab"] {\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  background: #0b1f3a !important;\n}\n\n.gradio-container .tab-nav button.selected,\n.gradio-container button[role="tab"][aria-selected="true"] {\n  color: #061426 !important;\n  -webkit-text-fill-color: #061426 !important;\n  background: #ffffff !important;\n}\n\n/* Markdown and HTML */\n.gradio-container .prose,\n.gradio-container .prose *,\n.gradio-container .markdown,\n.gradio-container .markdown *,\n.gradio-container .html-container,\n.gradio-container .html-container *,\n.gradio-container .gr-html,\n.gradio-container .gr-html * {\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n}\n\n/* Placeholder text */\n.gradio-container textarea::placeholder,\n.gradio-container input::placeholder {\n  color: #dbeafe !important;\n  -webkit-text-fill-color: #dbeafe !important;\n  opacity: 1 !important;\n}\n\n/* Selection */\n.gradio-container ::selection {\n  background: #2563eb !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n}\n\n/* Warning / info panels */\n.gradio-container .warning,\n.gradio-container .warning *,\n.gradio-container .error,\n.gradio-container .error *,\n.gradio-container .info,\n.gradio-container .info * {\n  background: #071629 !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n}\n\n/* Prosody custom sections */\n.prosody-card,\n.prosody-card *,\n.prosody-panel,\n.prosody-panel *,\n.voice-pattern,\n.voice-pattern *,\n.evidence-card,\n.evidence-card *,\n.summary-card,\n.summary-card * {\n  background-color: transparent !important;\n  color: #ffffff !important;\n  -webkit-text-fill-color: #ffffff !important;\n  opacity: 1 !important;\n}\n'

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

try:
    from .prosody_diagnostics import prosody_diagnostics
except Exception:
    prosody_diagnostics = None


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


def compute_prosody(audio_path: str, sample_rate: int) -> dict:
    if prosody_diagnostics is None:
        return {
            "available": False,
            "summary": "Prosody diagnostics are unavailable because the prosody module could not be imported.",
            "metrics": {},
            "interpretation": [],
            "warnings": [],
        }
    try:
        return prosody_diagnostics(audio_path, target_sr=sample_rate)
    except Exception as exc:
        return {
            "available": False,
            "summary": f"Prosody diagnostics failed: {exc}",
            "metrics": {},
            "interpretation": [],
            "warnings": [str(exc)],
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


def make_explanation(report: dict, prosody: dict | None = None) -> str:
    fake_prob = float(report.get("fake_probability", 0.0))
    attack_type = str(report.get("attack_type", "unknown"))
    decision = str(report.get("decision", "unknown"))
    suspicious_segments = report.get("suspicious_segments", [])
    warnings = report.get("decision_gate", {}).get("warnings", [])

    if beginner_friendly_explanation_plus is not None:
        base = beginner_friendly_explanation_plus(
            fake_prob=fake_prob,
            attack_type=attack_type,
            suspicious_segments=suspicious_segments,
            safe_decision=decision,
            warnings=warnings,
        )
    elif "real" in decision.lower():
        base = (
            "The system gives this audio a low fake score, so it is treated as likely real. "
            "The detected sound patterns are closer to real human speech than to generated speech."
        )
    elif "fake" in decision.lower():
        base = (
            "The system gives this audio a high fake score, so it is treated as likely fake. "
            "The detected sound patterns are similar to generated or manipulated speech."
        )
    else:
        base = (
            "The system is not fully confident. The result should be reviewed by a human before making a final decision."
        )

    if prosody and prosody.get("summary"):
        base += "\n\nProsody / tonality support: " + str(prosody.get("summary"))

    return base


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
        <div class="evidence-row"><span>Model score</span><b>{fake_prob:.4f}</b></div>
        <div class="evidence-row"><span>Attack-type clue</span><b>{html.escape(attack_type)}</b></div>
        <div class="evidence-row"><span>Strongest suspicious region</span><b>{top_segment_text}</b></div>
        <div class="evidence-row"><span>Reliability notes</span><b>{warning_text}</b></div>
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
        warning_block = f"<div class='warning-box'><b>Reliability Warnings</b><ul>{warnings_html}</ul></div>"
    else:
        warning_block = "<div class='ok-box'><b>No major audio-quality warnings detected.</b></div>"

    return f"""
    <div class="quality-panel">
        <h3>Audio Quality Diagnostics</h3>
        <div class="quality-grid">{metric_items}</div>
        {warning_block}
    </div>
    """


def prosody_html(prosody: dict) -> str:
    metrics = prosody.get("metrics", {}) or {}
    interpretation = prosody.get("interpretation", []) or []
    warnings = prosody.get("warnings", []) or []
    summary = html.escape(str(prosody.get("summary", "No prosody summary available.")))

    key_order = [
        "pitch_range_semitones",
        "pitch_std_semitones",
        "median_pitch_step_semitones",
        "energy_variation",
        "silence_ratio",
        "pause_like_gap_count",
        "voiced_fraction",
        "duration_sec",
    ]

    metric_items = ""
    for key in key_order:
        if key not in metrics:
            continue
        label = html.escape(key.replace("_", " ").title())
        value = metrics[key]
        if isinstance(value, float):
            value_text = f"{value:.3f}"
        else:
            value_text = str(value)
        metric_items += f"<div class='quality-item'><span>{label}</span><b>{html.escape(value_text)}</b></div>"

    if not metric_items:
        metric_items = "<div class='empty-message'>No prosody metrics available.</div>"

    interpretation_html = "".join(
        f"<li style='color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; opacity:1 !important; margin-bottom:8px;'>{html.escape(str(x))}</li>"
        for x in interpretation
    )
    if not interpretation_html:
        interpretation_html = "<li style='color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; opacity:1 !important;'>No interpretation available.</li>"

    warning_html = ""
    if warnings:
        warning_html = "<div class='warning-box'><b>Prosody Warnings</b><ul>" + "".join(
            f"<li>{html.escape(str(w))}</li>" for w in warnings
        ) + "</ul></div>"

    return f"""
    <div class="quality-panel">
        <h3>Prosody / Tonality Diagnostics</h3>
        <div class="ok-box"><b>Summary:</b> {summary}</div>
        <div class="quality-grid" style="margin-top:14px;">{metric_items}</div>
        <div class="evidence-panel voice-pattern-interpretation" style="margin-top:14px; background:#0b1f3a; color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;">
            <h3 style="color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;">Voice-Pattern Interpretation</h3>
            <ul style="color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;">{interpretation_html}</ul>
        </div>
        {warning_html}
        <div class="empty-message" style="margin-top:14px;">
            These diagnostics support the explanation only. They do not replace the AASIST fake probability and should not be treated as proof.
        </div>
    </div>
    """



def black_output_box(title: str, text: str) -> str:
    safe_title = html.escape(str(title))
    safe_text = html.escape("" if text is None else str(text))
    return f"""
    <div style="
        background:#000000;
        color:#ffffff;
        border:1px solid rgba(255,255,255,0.35);
        border-radius:14px;
        padding:18px 20px;
        min-height:120px;
        font-family:Inter, Arial, sans-serif;
        font-size:15px;
        line-height:1.55;
        white-space:pre-wrap;
        overflow:auto;
    ">
        <div style="font-weight:900; font-size:16px; margin-bottom:12px; color:#ffffff;">{safe_title}</div>
        <div style="color:#ffffff;">{safe_text}</div>
    </div>
    """


def black_json_box(title: str, obj) -> str:
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        text = str(obj)
    safe_title = html.escape(str(title))
    safe_text = html.escape(text)
    return f"""
    <div style="
        background:#000000;
        color:#ffffff;
        border:1px solid rgba(255,255,255,0.35);
        border-radius:14px;
        padding:18px 20px;
        min-height:220px;
        font-family:Consolas, 'Courier New', monospace;
        font-size:14px;
        line-height:1.45;
        white-space:pre;
        overflow:auto;
    ">
        <div style="font-family:Inter, Arial, sans-serif; font-weight:900; font-size:16px; margin-bottom:12px; color:#ffffff;">{safe_title}</div>
        <pre style="
            margin:0;
            background:#000000;
            color:#ffffff;
            white-space:pre-wrap;
            font-family:Consolas, 'Courier New', monospace;
        ">{safe_text}</pre>
    </div>
    """


def create_demo(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = load_model(build_model_args(args), device)

    def analyze(audio_path):
        if audio_path is None:
            empty = "<div class='empty-message'>Upload an audio file to begin analysis.</div>"
            return (empty, empty, "Upload an audio file to generate a simple explanation.", "[]", "{}", empty, empty, "")

        quality = compute_quality(audio_path, args.sample_rate)
        prosody = compute_prosody(audio_path, args.sample_rate)

        raw_report = run_auralguard(
            audio_path,
            model,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            device=device,
        )

        report = apply_safe_decision(raw_report, quality)
        report["prosody_report"] = prosody
        report["beginner_explanation"] = make_explanation(report, prosody)
        report["quality_report"] = quality

        export = {
            "final_auralguard_result": report,
            "quality_report": quality,
            "prosody_report": prosody,
        }

        report_path = Path("results") / "demo_professional_last_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(export, indent=2), encoding="utf-8")

        suspicious_segments_json = json.dumps(
            json.dumps(report.get("suspicious_segments", []), indent=2),
            indent=2,
            ensure_ascii=False,
        )
        evidence_packet_json = json.dumps(
            json.dumps(report.get("evidence", {}), indent=2),
            indent=2,
            ensure_ascii=False,
        )

        return (
            main_decision_html(report),
            evidence_summary_html(report),
            black_output_box("Simple explanation", report.get("beginner_explanation", "")),
            black_json_box("Suspicious timestamp regions", report.get("suspicious_segments", [])),
            black_json_box("Evidence packet", report.get("evidence", {})),
            quality_html(quality),
            prosody_html(prosody),
            black_output_box("Saved JSON report path", str(report_path)),
        )

    css = """
    :root { --bg-dark:#07111f; --bg-mid:#0b1b32; --panel:rgba(14,27,49,.95); --panel-soft:rgba(22,39,68,.95); --line:rgba(131,183,255,.26); --white:#ffffff; --blue:#4da3ff; --cyan:#28e0e8; --green:#39d98a; --yellow:#ffd166; --red:#ff5c7a; }
    body, .gradio-container { background: radial-gradient(circle at top left, rgba(40,224,232,.16), transparent 30%), radial-gradient(circle at top right, rgba(77,163,255,.16), transparent 32%), linear-gradient(135deg,var(--bg-dark),var(--bg-mid)) !important; color:var(--white)!important; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important; }
    .gradio-container { max-width:1180px!important; }
    .main-header { border:1px solid var(--line); background:linear-gradient(135deg,rgba(17,40,72,.97),rgba(8,17,31,.98)); border-radius:22px; padding:28px 30px; box-shadow:0 24px 70px rgba(0,0,0,.36); margin-bottom:18px; }
    .eyebrow { color:var(--cyan); font-size:13px; letter-spacing:.16em; text-transform:uppercase; font-weight:900; margin-bottom:10px; }
    .main-title { font-size:42px; line-height:1.05; margin:0; font-weight:950; color:var(--white); }
    .subtitle { font-size:16px; color:var(--white); margin-top:12px; max-width:860px; opacity:.92; }
    .feature-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
    .feature-chip { border:1px solid rgba(77,163,255,.28); background:rgba(77,163,255,.12); color:var(--white); border-radius:999px; padding:7px 11px; font-size:13px; font-weight:800; }
    .input-panel { border:1px solid var(--line)!important; background:var(--panel)!important; border-radius:20px!important; box-shadow:0 18px 45px rgba(0,0,0,.24)!important; padding:16px!important; }
    .decision-panel { border-radius:24px; padding:26px; border:1px solid var(--line); background:linear-gradient(135deg,rgba(19,38,68,.98),rgba(8,17,31,.98)); box-shadow:0 24px 70px rgba(0,0,0,.35); }
    .decision-panel.real { border-color:rgba(57,217,138,.65); } .decision-panel.fake { border-color:rgba(255,92,122,.68); } .decision-panel.review { border-color:rgba(255,209,102,.65); }
    .panel-label { color:var(--cyan); font-size:12px; font-weight:950; letter-spacing:.18em; text-transform:uppercase; }
    .decision-text { font-size:48px; font-weight:950; margin-top:8px; letter-spacing:-.03em; color:var(--white); }
    .decision-panel.real .decision-text { color:var(--green); } .decision-panel.fake .decision-text { color:var(--red); } .decision-panel.review .decision-text { color:var(--yellow); }
    .decision-description { color:var(--white); opacity:.92; margin-top:8px; font-size:15px; }
    .metric-grid, .quality-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:20px; }
    .metric-card, .quality-item, .evidence-row { background:rgba(255,255,255,.065); border:1px solid rgba(255,255,255,.10); border-radius:16px; padding:14px; }
    .metric-card span, .quality-item span, .evidence-row span { display:block; color:var(--white); opacity:.88; font-size:12px; text-transform:uppercase; letter-spacing:.08em; font-weight:850; }
    .metric-card b, .quality-item b, .evidence-row b { display:block; margin-top:7px; color:var(--white); font-size:18px; line-height:1.4; }
    .metric-card b { font-size:25px; }
    .evidence-panel, .quality-panel { border:1px solid var(--line); background:var(--panel-soft); border-radius:20px; padding:20px; box-shadow:0 18px 45px rgba(0,0,0,.24); color:var(--white); }
    .evidence-panel h3, .quality-panel h3 { color:var(--white); margin-top:0; margin-bottom:16px; font-size:23px; }
    .warning-box { border:1px solid rgba(255,209,102,.55); background:rgba(255,209,102,.12); border-radius:14px; padding:13px; color:var(--white); margin-top:14px; }
    .ok-box { border:1px solid rgba(57,217,138,.45); background:rgba(57,217,138,.12); border-radius:14px; padding:13px; color:var(--white); }
    .empty-message { border:1px dashed rgba(230,238,252,.42); border-radius:18px; padding:18px; color:var(--white); background:rgba(255,255,255,.045); }
    .prose, .markdown, label, .wrap, .json-holder, .tab-nav, .tabitem, .gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container p, .gradio-container span, .gradio-container label { color:var(--white)!important; }

    /* Final readability fix: black code boxes with white text */
    #evidence-json-box,
    #suspicious-json-box,
    #simple-explanation-box,
    #export-path-box {
        background: #000000 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,.35) !important;
        border-radius: 12px !important;
    }

    #evidence-json-box *,
    #suspicious-json-box *,
    #simple-explanation-box *,
    #export-path-box * {
        background: #000000 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    #evidence-json-box textarea,
    #suspicious-json-box textarea,
    #simple-explanation-box textarea,
    #export-path-box textarea,
    #evidence-json-box pre,
    #suspicious-json-box pre,
    #evidence-json-box code,
    #suspicious-json-box code,
    #evidence-json-box .cm-editor,
    #suspicious-json-box .cm-editor,
    #evidence-json-box .cm-content,
    #suspicious-json-box .cm-content,
    #evidence-json-box .cm-line,
    #suspicious-json-box .cm-line {
        background: #000000 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
        opacity: 1 !important;
    }

    .voice-pattern-interpretation,
    .voice-pattern-interpretation *,
    .warning-box,
    .warning-box *,
    .ok-box,
    .ok-box * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    button.primary { background:linear-gradient(90deg,var(--blue),var(--cyan))!important; color:#03101f!important; border:none!important; font-weight:950!important; border-radius:14px!important; box-shadow:0 12px 35px rgba(40,224,232,.26)!important; }
    @media (max-width:900px) { .metric-grid, .quality-grid { grid-template-columns:1fr; } .decision-text { font-size:38px; } }
    

/* FINAL TEXTBOX READABILITY FIX
   Uses Textbox instead of Gradio JSON/Code viewers because those can keep white panels
   or hidden syntax-highlighting colors in some browsers. */
#simple-explanation-box,
#suspicious-json-box,
#evidence-json-box,
#export-path-box {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 12px !important;
}

#simple-explanation-box *,
#suspicious-json-box *,
#evidence-json-box *,
#export-path-box * {
    background: #000000 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

#simple-explanation-box textarea,
#suspicious-json-box textarea,
#evidence-json-box textarea,
#export-path-box textarea {
    background: #000000 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
    opacity: 1 !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    font-family: Consolas, "Courier New", monospace !important;
}

#simple-explanation-box label,
#suspicious-json-box label,
#evidence-json-box label,
#export-path-box label {
    background: #111827 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 800 !important;
}

/* Prosody interpretation and warning text */
.evidence-panel ul,
.evidence-panel li,
.warning-box,
.warning-box *,
.ok-box,
.ok-box * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}



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

    with gr.Blocks(css=css, title="AuralGuard-AASIST++ Professional Dashboard", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
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
        """)

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["input-panel"]):
                gr.Markdown("### Upload Audio")
                audio_input = gr.Audio(type="filepath", label="Audio file")
                analyze_btn = gr.Button("Run Forensic Analysis", variant="primary")
                gr.Markdown("The dashboard returns the AuralGuard-AASIST++ decision, reliability notes, prosody/tonality clues, and forensic evidence for review.")
            with gr.Column(scale=2):
                final_decision = gr.HTML()

        with gr.Tab("Evidence Summary"):
            evidence_summary = gr.HTML()
        with gr.Tab("Beginner Explanation"):
            beginner_explanation = gr.HTML()
        with gr.Tab("Prosody / Tonality Diagnostics"):
            prosody_report = gr.HTML()
        with gr.Tab("Suspicious Timestamp Evidence"):
            suspicious_segments = gr.HTML()
        with gr.Tab("Evidence Packet"):
            evidence_packet = gr.HTML()
        with gr.Tab("Audio Quality Diagnostics"):
            quality_report = gr.HTML()
        with gr.Tab("Export"):
            export_path = gr.HTML()

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
                prosody_report,
                export_path,
            ],
        )

        gr.HTML("""
            <div style="margin-top:18px; color:#ffffff; opacity:.85; font-size:13px; text-align:center;">
                This dashboard provides forensic decision support, not legal proof. Prosody/tonality diagnostics are supporting clues only.
            </div>
        """)

    return demo


def main():
    args = parse_args()
    demo = create_demo(args)
    demo.launch(share=args.share, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
