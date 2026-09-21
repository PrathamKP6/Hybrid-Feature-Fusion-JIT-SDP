# ============================================================
# LLM FEATURE EXTRACTION — PRODUCTION V1
# 59,996 COMMITS
# Qwen3-4B + 4-bit NF4 + BF16
# ============================================================

import os
import csv
import json
import time
import gc

import torch
import bitsandbytes as bnb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
# 1. PATHS / CONFIGURATION
# ============================================================

INPUT_CSV = "/home/jovyan/llm_input_59996.csv"

OUTPUT_DIR = "/home/jovyan/llm_features_v1"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "llm_features.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.txt")
FAILED_CSV = os.path.join(OUTPUT_DIR, "failed_rows.csv")
METADATA_FILE = os.path.join(OUTPUT_DIR, "run_metadata.json")

MODEL_NAME = "Qwen/Qwen3-4B"
SCHEMA_VERSION = "v1"

MAX_LENGTH = 2048

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. OUTPUT SCHEMA
# ============================================================

OUTPUT_FIELDS = [
    "commit_id",

    "intent",
    "intent_confidence",
    "intent_margin",

    "change",
    "change_confidence",
    "change_margin",

    "risk",
    "risk_confidence",
    "risk_margin",

    "complexity",
    "complexity_confidence",
    "complexity_margin",

    "scope",
    "scope_confidence",
    "scope_margin",

    "test",
    "test_confidence",
    "test_margin",

    "security",
    "security_confidence",
    "security_margin",

    "model",
    "schema_version",
]

FIELDS = [
    "intent",
    "change",
    "risk",
    "complexity",
    "scope",
    "test",
    "security",
]

# ============================================================
# 3. LABEL DEFINITIONS
# ============================================================

LABELS = {

    "intent": [
        "BF",   # Bug Fix
        "FT",   # Feature
        "RF",   # Refactoring
        "DC",   # Documentation
        "TS",   # Testing
        "BL",   # Build/Dependency
        "PF",   # Performance
        "SC",   # Security
        "OT",   # Other
    ],

    "change": [
        "LG",   # Logic
        "AP",   # API
        "DA",   # Data
        "CF",   # Configuration
        "UI",   # User Interface
        "DP",   # Dependency
        "TS",   # Test
        "DC",   # Documentation
        "BL",   # Build/Release
        "OT",   # Other
    ],

    "risk": [
        "L",
        "M",
        "H",
    ],

    "complexity": [
        "L",
        "M",
        "H",
    ],

    "scope": [
        "L",   # Local
        "F",   # Focused
        "M",   # Multi-component
        "C",   # Cross-cutting
    ],

    "test": [
        "N",   # None/minimal
        "L",   # Low
        "M",   # Medium
        "H",   # High
    ],

    "security": [
        "N",   # None
        "L",   # Low
        "M",   # Medium
        "H",   # High
    ],
}

# ============================================================
# 4. FIELD-SPECIFIC INSTRUCTIONS
# ============================================================

FIELD_INSTRUCTIONS = {

    "intent": """
Classify WHY this commit was made.

BF = Bug Fix
FT = Feature
RF = Refactoring
DC = Documentation
TS = Testing
BL = Build/Dependency
PF = Performance
SC = Security
OT = Other
""",

    "change": """
Classify WHAT the commit changes.

LG = Logic
AP = API
DA = Data
CF = Configuration
UI = User Interface
DP = Dependency
TS = Test
DC = Documentation
BL = Build/Release
OT = Other
""",

    "risk": """
Classify overall defect-introduction RISK.

L = Low
M = Medium
H = High
""",

    "complexity": """
Classify technical COMPLEXITY.

L = Low
M = Medium
H = High
""",

    "scope": """
Classify CHANGE SCOPE.

L = Local
F = Focused
M = Multi-component
C = Cross-cutting
""",

    "test": """
Classify TESTING LEVEL introduced/required by the change.

N = None/minimal
L = Low
M = Medium
H = High
""",

    "security": """
Classify SECURITY relevance.

N = None
L = Low
M = Medium
H = High
""",
}

# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# ============================================================
# 6. LOAD QWEN3-4B
#    4-BIT NF4
#    BF16 COMPUTE
#    DOUBLE QUANTIZATION OFF
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_storage=torch.uint8,
)

model_singleq = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": "cuda:0"},
)

model_singleq.eval()

# ============================================================
# 7. VERIFY MODEL CONFIGURATION
# ============================================================

layer = next(
    m for m in model_singleq.modules()
    if isinstance(m, bnb.nn.Linear4bit)
)

qs = layer.weight.quant_state

assert layer.compute_dtype == torch.bfloat16
assert qs.quant_type == "nf4"
assert qs.nested is False

print("=" * 70)
print("PRODUCTION MODEL VERIFIED")
print("=" * 70)
print("Model          :", MODEL_NAME)
print("Quantization   : NF4")
print("Compute dtype  :", layer.compute_dtype)
print("Double quant   :", qs.nested)
print("Max length     :", MAX_LENGTH)
print("GPU            :", torch.cuda.get_device_name(0))
print(
    "GPU memory     :",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

# ============================================================
# 8. CONVERT LABELS TO TOKEN IDS
# ============================================================

LABEL_TOKEN_IDS = {}

for field, labels in LABELS.items():

    LABEL_TOKEN_IDS[field] = {}

    for label in labels:

        token_ids = tokenizer.encode(
            label,
            add_special_tokens=False
        )

        # Production schema assumes each label is one token.
        if len(token_ids) != 1:
            raise ValueError(
                f"Label {label} for field {field} "
                f"is not a single token: {token_ids}"
            )

        LABEL_TOKEN_IDS[field][label] = token_ids[0]

print("\nCandidate token IDs:")

for field, mapping in LABEL_TOKEN_IDS.items():
    print(field, mapping)

# ============================================================
# 9. PROMPT BUILDER
# ============================================================

def build_prompt(message, diff, field):

    return f"""You are classifying a software commit.

{FIELD_INSTRUCTIONS[field]}

Use BOTH the commit message and diff.
The diff is stronger evidence when message and implementation disagree.

Commit message:
{message}

Diff:
{diff}

Choose exactly one label from the allowed labels.
Answer with the label only.

Answer:"""

# ============================================================
# 10. PRODUCTION SCORER
# ============================================================

@torch.inference_mode()
def score_field_production(message, diff, field):

    prompt = build_prompt(
        message,
        diff,
        field
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        k: v.to("cuda:0")
        for k, v in inputs.items()
    }

    outputs = model_singleq(
        **inputs,
        use_cache=False,
    )

    # Logits for the token immediately after the prompt
    logits = outputs.logits[0, -1, :]

    labels = LABELS[field]

    token_ids = [
        LABEL_TOKEN_IDS[field][x]
        for x in labels
    ]

    # Keep only logits corresponding to valid labels
    candidate_logits = logits[token_ids]

    # Probability among allowed labels only
    probs = torch.softmax(
        candidate_logits.float(),
        dim=0,
    )

    # Best candidate
    best_idx = torch.argmax(probs).item()

    label = labels[best_idx]

    # Confidence
    confidence = float(
        probs[best_idx].item() * 100
    )

    # Top-1 minus top-2 probability
    sorted_probs = torch.sort(
        probs,
        descending=True,
    ).values

    margin = float(
        (sorted_probs[0] - sorted_probs[1]).item() * 100
    )

    return label, confidence, margin

# ============================================================
# 11. LOAD EXISTING OUTPUT FOR RESUME
# ============================================================

completed_ids = set()

if os.path.exists(OUTPUT_CSV):

    print(
        "\nExisting output found. "
        "Reading completed commit IDs..."
    )

    with open(
        OUTPUT_CSV,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            commit_id = row.get("commit_id")

            if commit_id:
                completed_ids.add(commit_id)

print(
    "Already completed:",
    len(completed_ids)
)

# ============================================================
# 12. CHECKPOINT
# ============================================================

checkpoint = 0

if os.path.exists(CHECKPOINT_FILE):

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            checkpoint = int(
                f.read().strip()
            )

    except Exception:

        checkpoint = 0

print(
    "Checkpoint:",
    checkpoint
)

# ============================================================
# 13. OUTPUT FILE INITIALIZATION
# ============================================================

output_exists = (
    os.path.exists(OUTPUT_CSV)
    and os.path.getsize(OUTPUT_CSV) > 0
)

output_file = open(
    OUTPUT_CSV,
    "a",
    encoding="utf-8",
    newline="",
    buffering=1,
)

output_writer = csv.DictWriter(
    output_file,
    fieldnames=OUTPUT_FIELDS,
)

if not output_exists:

    output_writer.writeheader()
    output_file.flush()

# ============================================================
# 14. FAILED ROW FILE
# ============================================================

failed_exists = (
    os.path.exists(FAILED_CSV)
    and os.path.getsize(FAILED_CSV) > 0
)

failed_file = open(
    FAILED_CSV,
    "a",
    encoding="utf-8",
    newline="",
    buffering=1,
)

failed_writer = csv.DictWriter(
    failed_file,
    fieldnames=[
        "row_number",
        "commit_id",
        "error",
    ],
)

if not failed_exists:

    failed_writer.writeheader()
    failed_file.flush()

# ============================================================
# 15. METADATA
# ============================================================

metadata = {
    "model": MODEL_NAME,
    "schema_version": SCHEMA_VERSION,
    "quantization": "NF4",
    "compute_dtype": "BF16",
    "double_quantization": False,
    "max_length": MAX_LENGTH,
    "fields": FIELDS,
    "input_csv": INPUT_CSV,
    "output_csv": OUTPUT_CSV,
    "resume_enabled": True,
    "checkpoint_file": CHECKPOINT_FILE,
}

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )

# ============================================================
# 16. PRODUCTION EXTRACTION LOOP
# ============================================================

processed = 0
skipped = 0
failed = 0

start_time = time.time()
last_report = start_time

print("\n" + "=" * 70)
print("STARTING PRODUCTION EXTRACTION")
print("=" * 70)

with open(
    INPUT_CSV,
    "r",
    encoding="utf-8",
    newline="",
) as input_file:

    reader = csv.DictReader(input_file)

    for row_number, row in enumerate(
        reader,
        start=1
    ):

        commit_id = row["commit_id"]

        # ----------------------------------------------------
        # Resume / skip already processed commits
        # ----------------------------------------------------

        if commit_id in completed_ids:

            skipped += 1
            continue

        try:

            message = (
                row.get(
                    "commit_message",
                    ""
                ) or ""
            )

            diff = (
                row.get(
                    "commit_diff",
                    ""
                ) or ""
            )

            # ------------------------------------------------
            # Score all seven fields
            # ------------------------------------------------

            results = {}

            for field in FIELDS:

                label, confidence, margin = (
                    score_field_production(
                        message,
                        diff,
                        field
                    )
                )

                results[field] = (
                    label,
                    confidence,
                    margin
                )

            # ------------------------------------------------
            # Construct final row
            # ------------------------------------------------

            output_row = {
                "commit_id": commit_id,

                "intent": results["intent"][0],
                "intent_confidence":
                    round(results["intent"][1], 4),
                "intent_margin":
                    round(results["intent"][2], 4),

                "change": results["change"][0],
                "change_confidence":
                    round(results["change"][1], 4),
                "change_margin":
                    round(results["change"][2], 4),

                "risk": results["risk"][0],
                "risk_confidence":
                    round(results["risk"][1], 4),
                "risk_margin":
                    round(results["risk"][2], 4),

                "complexity":
                    results["complexity"][0],
                "complexity_confidence":
                    round(results["complexity"][1], 4),
                "complexity_margin":
                    round(results["complexity"][2], 4),

                "scope":
                    results["scope"][0],
                "scope_confidence":
                    round(results["scope"][1], 4),
                "scope_margin":
                    round(results["scope"][2], 4),

                "test":
                    results["test"][0],
                "test_confidence":
                    round(results["test"][1], 4),
                "test_margin":
                    round(results["test"][2], 4),

                "security":
                    results["security"][0],
                "security_confidence":
                    round(results["security"][1], 4),
                "security_margin":
                    round(results["security"][2], 4),

                "model":
                    "Qwen3-4B",

                "schema_version":
                    "v1",
            }

            # ------------------------------------------------
            # Write output
            # ------------------------------------------------

            output_writer.writerow(
                output_row
            )

            output_file.flush()

            completed_ids.add(
                commit_id
            )

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            with open(
                CHECKPOINT_FILE,
                "w",
                encoding="utf-8",
            ) as checkpoint_file:

                checkpoint_file.write(
                    str(row_number)
                )

            processed += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            now = time.time()

            if now - last_report >= 10:

                elapsed = (
                    now - start_time
                )

                rate = (
                    processed / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    59996 -
                    len(completed_ids)
                )

                eta_hours = (
                    remaining /
                    rate /
                    3600
                    if rate > 0
                    else float("inf")
                )

                print(
                    f"Processed: {processed:,} | "
                    f"Skipped: {skipped:,} | "
                    f"Failed: {failed:,} | "
                    f"Rate: {rate:.2f} commits/s | "
                    f"ETA: {eta_hours:.2f} h"
                )

                last_report = now

        except Exception as e:

            failed += 1

            failed_writer.writerow({
                "row_number": row_number,
                "commit_id": commit_id,
                "error": repr(e),
            })

            failed_file.flush()

            print(
                f"FAILED: {commit_id} -> {repr(e)}"
            )

# ============================================================
# 17. CLEANUP
# ============================================================

output_file.close()
failed_file.close()

gc.collect()
torch.cuda.empty_cache()

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("PRODUCTION EXTRACTION FINISHED")
print("=" * 70)

print("Processed:", processed)
print("Skipped  :", skipped)
print("Failed   :", failed)
print("Output   :", OUTPUT_CSV)
print("Checkpoint:", CHECKPOINT_FILE)
print("Failed rows:", FAILED_CSV)
print(
    "Elapsed:",
    round(elapsed / 3600, 2),
    "hours"
)