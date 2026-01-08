import subprocess
import sys
from pathlib import Path


def test_auxiliary_includes_dealer_trace():
    # Run the data_visual script to generate chart.html from the provided preprocessed JSON
    script = Path('engine/datasets_ml/data_visual.py').resolve()
    json_path = Path('engine/datasets_ml/preprocessed_3034.json').resolve()

    # Execute the script (it writes chart.html next to the JSON file)
    subprocess.run([sys.executable, str(script), str(json_path)], check=True)

    out_fp = Path('engine/datasets_ml/chart.html')
    assert out_fp.exists(), f"Expected chart.html at {out_fp}"

    content = out_fp.read_text(encoding='utf-8')

    # Ensure dealer and investment-trust trace names appear in the generated HTML
    assert '自營商' in content, "Dealer trace name '自營商' not found in generated HTML"
    assert '投信' in content, "Investment trust trace '投信' not found in generated HTML"