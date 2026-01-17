from pathlib import Path

ROBOT_IP = "192.168.253.102"
PORT = 63352
GEMINI_336_NUMBER = "CP9JA530008V"

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data"
DEMO_PATH = DATA_PATH / "demos"
SAMPLE_PATH = DATA_PATH / "samples"
VIS_PATH = DATA_PATH / "visualizations"