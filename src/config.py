from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "report"
PROCESSED_DIR = OUTPUT_DIR / "processed"

EXPRESSION_FILE = RAW_DATA_DIR / "expression.csv"
METHYLATION_FILE = RAW_DATA_DIR / "methylation.csv"
CLINICAL_FILE = RAW_DATA_DIR / "clinical.csv"

LABEL_COLUMN_CANDIDATES = ["label", "Label", "class", "Class", "status"]
AGE_COLUMN_CANDIDATES = ["age", "Age", "patient_age", "PatientAge"]
SAMPLE_ID_COLUMN_CANDIDATES = ["sample_id", "SampleID", "sample", "Sample", "id", "ID"]
