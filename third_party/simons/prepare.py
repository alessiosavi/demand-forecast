"""Apply the reviewed patch set only to the exact, unmodified upstream pin."""

import argparse
import subprocess
from pathlib import Path


def prepare(tree: Path, bundle: Path) -> None:
    """Check revision and all patches before changing any upstream file."""
    pin = (bundle / "PINNED").read_text().strip()
    head = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    if head != pin:
        raise ValueError(f"upstream pin mismatch: expected {pin}, found {head}")
    subprocess.run(
        ["git", "-C", str(tree), "diff", "HEAD", "--exit-code", "--", "src", "configs"],
        check=True,
        capture_output=True,
    )
    patches = sorted((bundle / "patches").glob("*.patch"))
    if not patches:
        raise ValueError("empty Simons patch set")
    command = ["git", "-C", str(tree), "apply"]
    paths = [str(p.resolve()) for p in patches]
    subprocess.run([*command, "--check", *paths], check=True)
    subprocess.run([*command, *paths], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    prepare(args.tree, args.bundle)
