# Identifile

Lightweight Python library to detect common file formats (compression, archive, and columnar storage) by inspecting file signatures and heuristics.

## Project

- Name: Identifile
- Version: 2.0.0
- Authors: Sai Niranjan Chitturi, Leela Sai Surya Veer Pedarla, Bhargav Dasari, Kavya Sri Punna, Vinay Appari, Chandralekha Alluri, Eknath Narravula, Pavan, Divya, OpenAI, xAI
- License: MIT

This repository contains a small utility to sniff file formats from a file path or a binary stream. It recognizes formats such as gzip, zstd, bzip2, lz4, xz, ZIP/7z/tar, Snappy variants, Brotli (heuristic), Parquet and ORC.

## Features

- Signature-based detection for common compressed and archived formats.
- Columnar format detection for Parquet and ORC.
- Work with file paths (`sniff_format`) or file-like binary streams (`sniff_stream`).
- Extension hints for formats that lack headers (e.g. raw Snappy files).
- Small and easily extensible signature table. Add custom signatures with `add_signature()`.

## Files

- `Identifile.py` — Main implementation (defines `Identifile`, `sniff_format`, `sniff_stream`, `add_signature`, and `SIGNATURES`).
- `testFormat.py` — Unit tests covering detection logic and edge-cases using `unittest`.

## Usage

Example (detect from file path):

```python
from Identifile import sniff_format

result = sniff_format('example.gz')
print(result.summary())
```

Example (detect from stream):

```python
from Identifile import sniff_stream
from io import BytesIO

stream = BytesIO(b'\x1f\x8b' + b'\x00'*100)
result = sniff_stream(stream)
print(result.format, result.confidence, result.evidence)
```

Add a custom signature:

```python
from Identifile import add_signature

add_signature('myfmt', {'start': [b'\xAA\xBB'], 'evidence': 'My custom format.'})
```

## API Summary

- Identifile(format: str, confidence: float, evidence: str, extra: dict)
	- Methods: `is_known()`, `is_compressed()`, `is_archive()`, `is_columnar()`, `summary()`, `metadata()`
- sniff_format(file_path: str, head_n: int = 64, tail_n: int = 64, use_extension_hint: bool = True) -> Identifile
- sniff_stream(stream: BinaryIO, head_n: int = 64, tail_n: int = 64, buffer_non_seekable: bool = True, extension_hint: Optional[str] = None) -> Identifile
- add_signature(format_name: str, signature: dict, overwrite: bool = False)

## Running Tests

The project uses Python's built-in `unittest`. From the project root run:

```powershell
python -m unittest testFormat.py
```

Notes:
- Tests in `testFormat.py` reference `Identifile` module and exercise both file and stream-based detection. Ensure the working directory includes `Identifile.py`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Authors and Contributors

- Your Name <your.email@example.com> — primary author. Update `Identifile.py` to set `__author__` and replace this entry.
- Tests and example usage were authored based on the implementation in `testFormat.py`.

Contributions are welcome. Please open issues or pull requests with improvements, additional signatures, or bug fixes.

## Contributing

- Fork the repository, create a feature branch, add tests for new behavior, and open a pull request.
- Keep signatures small and add clear evidence strings.

## Changes

- v0.1.2 — initial public snapshot (version string found in `Identifile.py`).

---

If you'd like, I can also update the `__author__` value in `Identifile.py` to match a specific name/email and add a short example script. Tell me how you'd like the author line to read.