#!/usr/bin/env python3
"""
Script to split a jupytext Python file into student and solution versions.

Usage:
    python split_exercises.py input_file.py

This will create:
    - input_file_student.py (exercises without solutions)
    - input_file_solution.py (exercises with solutions)
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def parse_cells(content: str) -> List[Tuple[Optional[str], str]]:
    """Parse the content into cells based on # %% markers.

    Returns list of (cell_type, cell_content) tuples.
    """
    # Split by cell markers, keeping the markers
    cell_pattern = r"(# %%(?:\s+\w+)?.*?\n)"
    parts = re.split(cell_pattern, content)

    cells = []
    i = 0

    # Handle potential content before first cell marker
    if parts[0].strip() and not parts[0].startswith("# %%"):
        cells.append((None, parts[0]))
        i = 1

    # Process the rest
    while i < len(parts) - 1:
        marker = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""

        # Extract cell type from marker
        cell_type_match = re.match(r"# %%\s*(\w+)?", marker)
        if cell_type_match:
            cell_type = cell_type_match.group(1)
        else:
            cell_type = None

        cells.append((cell_type, marker + content))
        i += 2

    return cells


def create_student_version(cells: List[Tuple[Optional[str], str]]) -> str:
    """Create student version by removing solution cells and adding TODO markers."""
    output = []
    skip_next_solution = False

    for i, (cell_type, content) in enumerate(cells):
        if cell_type == "[exercise]":
            # Add the exercise cell
            output.append(content)
            skip_next_solution = True
        elif cell_type == "[solution]":
            if skip_next_solution:
                # Replace solution with a placeholder
                output.append("# %% solution\n# TODO: Complete the exercise above\n")
                skip_next_solution = False
            else:
                # Standalone solution without preceding exercise
                output.append(content)
        else:
            # Regular cell - include as is
            output.append(content)
            skip_next_solution = False

    return "".join(output)


def create_solution_version(cells: List[Tuple[Optional[str], str]]) -> str:
    """Create solution version with all cells included."""
    return "".join(content for _, content in cells)


def process_file(input_path: Path) -> None:
    """Process the input file and create student/solution versions."""
    # Read input file
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse cells
    cells = parse_cells(content)

    # Create output filenames
    stem = input_path.stem
    suffix = input_path.suffix
    student_path = input_path.parent / f"{stem}_student{suffix}"
    solution_path = input_path.parent / f"{stem}_solution{suffix}"

    # Create student version
    student_content = create_student_version(cells)
    with open(student_path, "w", encoding="utf-8") as f:
        f.write(student_content)
    print(f"Created student version: {student_path}")

    # Create solution version
    solution_content = create_solution_version(cells)
    with open(solution_path, "w", encoding="utf-8") as f:
        f.write(solution_content)
    print(f"Created solution version: {solution_path}")

    # Print summary
    exercise_count = sum(1 for cell_type, _ in cells if cell_type == "exercise")
    solution_count = sum(1 for cell_type, _ in cells if cell_type == "solution")
    print(f"\nSummary:")
    print(f"  Total cells: {len(cells)}")
    print(f"  Exercise cells: {exercise_count}")
    print(f"  Solution cells: {solution_count}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python split_exercises.py input_file.py")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found")
        sys.exit(1)

    process_file(input_path)


if __name__ == "__main__":
    main()
