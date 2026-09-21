"""
Generate SHA-256 checksums for the publication/reproducibility artifacts.

The checksum file itself and all files outside the explicit publication
artifact allowlist are excluded.
"""

from pathlib import Path
import hashlib
import logging
import sys


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "data" / "public_release"
OUTPUT = PUBLIC_DIR / "SHA256SUMS.txt"
CHUNK_SIZE = 1024 * 1024


EXPECTED_ARTIFACTS = [
	"final_multilingual_jit_codebert_llm_59996_768d.csv",
	"final_multilingual_jit_codebert_llm_59996_768d.zip",
	"AUDIT_REPORT.txt",
	"FEATURE_DICTIONARY.csv",
	"FEATURE_DICTIONARY.md",
	"DATASET.md",
	"split_ids/train_ids.txt",
	"split_ids/validation_ids.txt",
	"split_ids/test_ids.txt",
	"split_ids/split_summary.txt",
]


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
	"""Return a file's SHA-256 digest using bounded binary reads."""
	digest = hashlib.sha256()
	with path.open("rb") as file:
		for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
			digest.update(chunk)
	return digest.hexdigest()


def main() -> int:
	logger.info("Generating publication SHA-256 checksums")

	relative_paths = sorted(EXPECTED_ARTIFACTS)
	missing = [
		relative_path
		for relative_path in relative_paths
		if not (PUBLIC_DIR / relative_path).is_file()
	]
	if missing:
		raise FileNotFoundError(
			"Required publication artifacts are missing:\n"
			+ "\n".join(f"- {relative_path}" for relative_path in missing)
		)

	checksum_lines = []
	for relative_path in relative_paths:
		path = PUBLIC_DIR / relative_path
		checksum = sha256_file(path)
		checksum_lines.append(f"{checksum}  {relative_path}")
		logger.info("Hashed %s: %s", relative_path, checksum)
		print(f"{relative_path}: {checksum}")

	OUTPUT.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

	logger.info("Wrote checksum file: %s", OUTPUT)
	logger.info("Hashed %d files.", len(checksum_lines))
	print(f"Output: {OUTPUT}")
	print(f"Files hashed: {len(checksum_lines)}")
	print(f"SUCCESS: generated {OUTPUT}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except Exception as exc:
		logger.error("FAILED: %s", exc)
		print(f"ERROR: {exc}", file=sys.stderr)
		raise SystemExit(1)
