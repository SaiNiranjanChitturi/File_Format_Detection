import unittest
from io import BytesIO
from typing import Dict
import os
import tempfile

# Assuming the library is saved as fileformatdetect.py and imported here.
# If it's in the same file, adjust accordingly.
from Identifile import add_signature, sniff_stream, sniff_format, SIGNATURES

class TestIdentifile(unittest.TestCase):
    """
    Unit tests for the file format detection library.
    Covers all supported formats, unknown cases, and extension hints.
    """

    def _create_stream(self, data: bytes) -> BytesIO:
        """Helper to create a seekable BytesIO stream."""
        return BytesIO(data)

    def _create_temp_file(self, data: bytes, extension: str = '') -> str:
        """Helper to create a temporary file for sniff_format testing."""
        fd, path = tempfile.mkstemp(suffix=extension)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        return path

    def _cleanup_temp_file(self, path: str):
        """Cleanup temporary file."""
        if os.path.exists(path):
            os.remove(path)

    def test_gzip(self):
        data = b'\x1f\x8b' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'gzip')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_compressed())
        self.assertEqual(det.evidence, "Starts with 1f 8b (gzip).")

    def test_zstd(self):
        data = b'\x28\xb5\x2f\xfd' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'zstd')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_compressed())
        self.assertEqual(det.evidence, "Starts with 28 b5 2f fd (zstd).")

    def test_bzip2(self):
        data = b'BZh' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'bzip2')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_compressed())
        self.assertEqual(det.evidence, "Starts with 'BZh' (bzip2).")

    def test_lz4_frame(self):
        data = b'\x04\x22\x4d\x18' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'lz4-frame')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_compressed())
        self.assertEqual(det.evidence, "Starts with 04 22 4d 18 (LZ4 frame).")

    def test_xz(self):
        data = b'\xfd7zXZ\x00' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'xz')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_compressed())
        self.assertEqual(det.evidence, "Starts with fd 37 7a 58 5a 00 (XZ).")

    def test_zip(self):
        for start in [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08']:
            data = start + b'\x00' * 100
            stream = self._create_stream(data)
            det = sniff_stream(stream)
            self.assertEqual(det.format, 'zip')
            self.assertEqual(det.confidence, 1.0)
            self.assertTrue(det.is_archive())
            self.assertEqual(det.evidence, "Starts with PK (ZIP container).")

    def test_7z(self):
        data = b'\x37\x7a\xbc\xaf\x27\x1c' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, '7z')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_archive())
        self.assertEqual(det.evidence, "Starts with 37 7a bc af 27 1c (7z archive).")

    def test_snappy_framed(self):
        data = b'\xff\x06\x00\x00sNaPpY' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'snappy-framed')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_compressed())
        self.assertEqual(det.evidence, "Starts with ff 06 00 00 'sNaPpY' (Snappy framed).")

    def test_snappy_snz(self):
        data = b'SNZ\x01' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'snappy-snz')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_compressed())
        self.assertEqual(det.evidence, "Starts with 'SNZ\\x01' (obsolete Snappy SNZ format).")

    def test_brotli(self):
        # Test with first byte 0x91
        data = b'\x91' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'brotli')
        self.assertEqual(det.confidence, 0.5)
        self.assertTrue(det.is_compressed())
        self.assertEqual(det.evidence, "First byte in range 0x91-0x9F (Brotli heuristic).")

        # Test upper range 0x9F
        data = b'\x9F' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'brotli')
        self.assertEqual(det.confidence, 0.5)

        # Negative test: outside range
        data = b'\x90' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'unknown')

    def test_parquet(self):
        data = b'PAR1' + b'\x00' * 100 + b'PAR1'
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'parquet')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_columnar())
        self.assertEqual(det.evidence, "Has 'PAR1' at start and end (Parquet).")

        # Negative: missing end
        data = b'PAR1' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'unknown')

    def test_orc(self):
        data = b'\x00' * 100 + b'ORC'
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'orc')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_columnar())
        self.assertEqual(det.evidence, "Tail contains 'ORC' in postscript (ORC).")

        # Negative: no 'ORC' in tail
        data = b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'unknown')

    def test_tar(self):
        # 'ustar\x00' at offset 257
        data = b'\x00' * 257 + b'ustar\x00' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'tar')
        self.assertEqual(det.confidence, 1.0)
        self.assertTrue(det.is_archive())
        self.assertEqual(det.evidence, "Has 'ustar' at offset 257 (TAR).")

        # Alternative magic 'ustar  '
        data = b'\x00' * 257 + b'ustar  ' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'tar')
        self.assertEqual(det.confidence, 1.0)

        # Negative: wrong position
        data = b'\x00' * 256 + b'ustar\x00' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'unknown')

    def test_snappy_raw_with_extension_hint(self):
        # No signature, rely on hint
        data = b'\x00' * 100  # Arbitrary data
        stream = self._create_stream(data)
        det = sniff_stream(stream, extension_hint='.snz')
        self.assertEqual(det.format, 'snappy-raw')
        self.assertEqual(det.confidence, 0.5)
        self.assertTrue(det.is_compressed())
        self.assertIn('Based on extension hint', det.evidence)

        # Test other extensions
        det = sniff_stream(stream, extension_hint='.snappy')
        self.assertEqual(det.format, 'snappy-raw')
        det = sniff_stream(stream, extension_hint='.sz')
        self.assertEqual(det.format, 'snappy-raw')

        # Without hint, unknown
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'unknown')

    def test_unknown(self):
        data = b'random_data_without_signature' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'unknown')
        self.assertEqual(det.confidence, 0.0)
        self.assertFalse(det.is_known())
        self.assertEqual(det.evidence, "No decisive signature found.")

    def test_sniff_format_with_file(self):
        # Test with a temp file for gzip
        data = b'\x1f\x8b' + b'\x00' * 100
        path = self._create_temp_file(data)
        try:
            det = sniff_format(path)
            self.assertEqual(det.format, 'gzip')
        finally:
            self._cleanup_temp_file(path)

        # Test extension hint with .snz
        data = b'\x00' * 100
        path = self._create_temp_file(data, '.snz')
        try:
            det = sniff_format(path)
            self.assertEqual(det.format, 'snappy-raw')
            self.assertEqual(det.confidence, 0.5)

            # Disable hint
            det = sniff_format(path, use_extension_hint=False)
            self.assertEqual(det.format, 'unknown')
        finally:
            self._cleanup_temp_file(path)

    def test_non_seekable_stream_partial(self):
        # Simulate non-seekable stream by reading head only
        class NonSeekableStream:
            def __init__(self, data):
                self.data = data
                self.pos = 0

            def read(self, n=-1):
                if n == -1:
                    result = self.data[self.pos:]
                    self.pos = len(self.data)
                else:
                    result = self.data[self.pos:self.pos + n]
                    self.pos += len(result)
                return result

            def seekable(self):
                return False

        # Test head-based format (gzip)
        data = b'\x1f\x8b' + b'\x00' * 100
        stream = NonSeekableStream(data)
        det = sniff_stream(stream, buffer_non_seekable=False)
        self.assertEqual(det.format, 'gzip')
        self.assertEqual(det.confidence, 0.8)  # Reduced for partial
        self.assertIn('partial detection', det.evidence)

        # Test tail-based format (orc) - should fail to unknown without buffering
        data = b'\x00' * 100 + b'ORC'
        stream = NonSeekableStream(data)
        det = sniff_stream(stream, buffer_non_seekable=False)
        self.assertEqual(det.format, 'unknown')

        # With buffering, should detect
        stream = NonSeekableStream(data)
        det = sniff_stream(stream, buffer_non_seekable=True)
        self.assertEqual(det.format, 'orc')
        self.assertEqual(det.confidence, 1.0)

    def test_add_signature(self):
        # Test adding a custom signature
        custom_sig = {
            "start": [b'\xAA\xBB'],
            "evidence": "Custom format start.",
        }
        add_signature('custom', custom_sig)

        data = b'\xAA\xBB' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'custom')
        self.assertEqual(det.confidence, 1.0)

        # Cleanup: remove custom
        del SIGNATURES['custom']

        # Test overwrite
        with self.assertRaises(ValueError):
            add_signature('gzip', custom_sig, overwrite=False)
        add_signature('gzip', custom_sig, overwrite=True)  # Overwrite temporarily
        SIGNATURES['gzip'] = {  # Restore original
            "start": [b"\x1f\x8b"],
            "evidence": "Starts with 1f 8b (gzip).",
        }

    def test_small_file(self):
        # Test file smaller than probes
        data = b'\x1f\x8b'  # Only 2 bytes
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'gzip')  # Still detects start

        # Tar with small size (less than 257 + 8)
        data = b'\x00' * 200
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.format, 'unknown')  # No tar slice

    def test_summary_and_metadata(self):
        data = b'\x1f\x8b' + b'\x00' * 100
        stream = self._create_stream(data)
        det = sniff_stream(stream)
        self.assertEqual(det.summary(), '[GZIP] confidence=1.00 – Starts with 1f 8b (gzip).')
        meta = det.metadata()
        self.assertEqual(meta['format'], 'gzip')
        self.assertTrue(meta['is_compressed'])
        self.assertFalse(meta['is_archive'])
        self.assertFalse(meta['is_columnar'])

if __name__ == '__main__':
    unittest.main()