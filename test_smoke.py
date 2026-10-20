"""Offline integration test, requires NumPy/SciPy only."""
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    (folder/'key.bin').write_bytes(b'example-test-key-1234567890')
    with wave.open(str(folder/'input.wav'), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        t = np.arange(16000 * 3)/16000
        f.writeframes((np.sin(t*2*np.pi*233)*6000).astype('<i2').tobytes())
    base = [sys.executable, str(root/'main.py'), '--key-file', str(folder/'key.bin')]
    subprocess.run(base + ['--role','tx','--input',str(folder/'input.wav'),'--out',str(folder/'tx')], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(base + ['--role','rx','--input',str(folder/'tx/tx_wire.wav'),'--out',str(folder/'rx')], check=True, stdout=subprocess.DEVNULL)
    import json
    stats = json.loads((folder/'rx/summary.json').read_text())
    assert stats['acquisitions'] == 1, stats
    assert stats['frames_ok'] == 150 and stats['frames_lost'] == 0, stats
    assert stats['pilot_verdict'] == 'LIKELY_OK', stats
    print('PASS: 150 physical packets, PN excluded from RX WAV, acquired, 0 lost, pilot verified heuristically')
