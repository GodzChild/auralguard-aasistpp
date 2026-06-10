from pathlib import Path

path = Path('src/demo_professional.py')
if not path.exists():
    raise FileNotFoundError('Could not find src/demo_professional.py. Run this from the project root.')

text = path.read_text(encoding='utf-8')

text = text.replace(
    'beginner_explanation = gr.Textbox(label="Simple explanation", lines=8)',
    'beginner_explanation = gr.Textbox(label="Simple explanation", lines=8, elem_id="simple-explanation-box")'
)

text = text.replace(
    'suspicious_segments = gr.JSON(label="Suspicious timestamp regions")',
    'suspicious_segments = gr.JSON(label="Suspicious timestamp regions", elem_id="suspicious-json-box")'
)

text = text.replace(
    'evidence_packet = gr.JSON(label="Evidence packet")',
    'evidence_packet = gr.JSON(label="Evidence packet", elem_id="evidence-json-box")'
)

css_patch = '''
    /* Readability fix: JSON outputs and Simple Explanation only */
    #evidence-json-box,
    #suspicious-json-box,
    #simple-explanation-box {
        background: #0a1424 !important;
        border: 1px solid rgba(131, 183, 255, 0.35) !important;
        border-radius: 16px !important;
    }

    #evidence-json-box *,
    #suspicious-json-box * {
        color: #f8fbff !important;
        background-color: transparent !important;
    }

    #evidence-json-box pre,
    #suspicious-json-box pre,
    #evidence-json-box code,
    #suspicious-json-box code,
    #evidence-json-box .json-holder,
    #suspicious-json-box .json-holder {
        background: #07111f !important;
        color: #f8fbff !important;
        border-radius: 12px !important;
    }

    #evidence-json-box span,
    #suspicious-json-box span,
    #evidence-json-box div,
    #suspicious-json-box div {
        color: #f8fbff !important;
    }

    #simple-explanation-box textarea,
    #simple-explanation-box input,
    #simple-explanation-box .wrap {
        background: #07111f !important;
        color: #f8fbff !important;
        border-color: rgba(131, 183, 255, 0.35) !important;
    }

    #simple-explanation-box label,
    #evidence-json-box label,
    #suspicious-json-box label {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

'''

if '/* Readability fix: JSON outputs and Simple Explanation only */' not in text:
    if '    button.primary {' in text:
        text = text.replace('    button.primary {', css_patch + '\n    button.primary {')
    else:
        text = text.replace('    """\n\n    with gr.Blocks', css_patch + '\n    """\n\n    with gr.Blocks')
else:
    print('Readability CSS already exists; skipping CSS injection.')

path.write_text(text, encoding='utf-8')
print('Patched src/demo_professional.py successfully.')