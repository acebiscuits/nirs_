#!/usr/bin/env python3
"""Offline validation: key stream, 64-PN sync, 3 listening branches, codec Opus optional.
No microphone, PulseAudio or key.bin is required. Uses a temporary test-only key.
"""
import argparse, json, os, subprocess, sys, tempfile, wave
from pathlib import Path
import numpy as np
from scipy.signal import correlate

ROOT=Path(__file__).resolve().parent
FS=16000

def audio_file(path, samples, rate=FS):
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(np.rint(np.clip(samples,-1,1)*32767).astype('<i2').tobytes())

def read_audio(path):
    with wave.open(str(path),'rb') as f:
        return np.frombuffer(f.readframes(f.getnframes()),'<i2').astype('float64')/32768

def run(role, key, input_path, out, cipher, op, pilot=0, plugin=None):
    cmd=[sys.executable, str(ROOT/'main.py'), role,'--key-file',str(key),'--input',str(input_path),
         '--out',str(out),'--cipher',cipher,'--operation',op,'--pilot-seconds',str(pilot)]
    if plugin:cmd+=['--plugin', plugin]
    subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
    return json.loads((out/'summary.json').read_text()) if role=='rx' else None

def metric(a,b):
    """Best aligned normalized correlation, excluding startup and tail."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    a=a[7200:-3200];b=b[7200:-3200]
    a-=a.mean();b-=b.mean()
    if len(a)<FS or len(b)<FS:return 0.0
    v=correlate(a,b,mode='full',method='fft')
    lag=np.arange(-len(b)+1,len(a))
    mask=np.abs(lag)<=3200
    denom=max(1e-12,float(np.linalg.norm(a)*np.linalg.norm(b)))
    return round(max(0.0,float(np.max(v[mask])/denom)),3)

def codec_opus(src,out):
    import shutil
    ff=shutil.which('ffmpeg')
    if not ff:return False
    opus=out.with_suffix('.opus')
    subprocess.run([ff,'-y','-loglevel','error','-i',str(src),'-ar','48000','-c:a','libopus','-b:a','96k',str(opus)],check=True,timeout=30)
    subprocess.run([ff,'-y','-loglevel','error','-i',str(opus),'-ac','1','-ar','20000','-c:a','pcm_s16le',str(out)],check=True,timeout=30)
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--codec',action='store_true',help='Also pass a wire WAV through Opus 96 kbps')
    args=ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        key=d/'key.bin';key.write_bytes(b'just-a-test-key-for-integration-123456')
        t=np.arange(FS*4)/FS
        # Pseudo-speech: voiced harmonics + pauses + rising and falling intensity.
        env=(0.6+0.4*np.sin(t*7)**2)*(np.mod(t,1.0)<0.75)
        x=env*(0.10*np.sin(2*np.pi*(190*t+7*t*t)) +0.035*np.sin(2*np.pi*390*t)+0.02*np.sin(2*np.pi*650*t))
        audio_file(d/'voice.wav',x)
        for cipher,op in [('stft','signflip'),('stft','shuffle'),('stft','invert'),('dct4','signflip'),('dct4','shuffle'),('dct4','invert')]:
            test=d/(cipher+'_'+op); test.mkdir()
            run('tx',key,d/'voice.wav',test/'tx',cipher,op)
            raw=run('rx',key,test/'tx/tx_wire.wav',test/'rx',cipher,op)
            assert raw['frames_ok']==200 and raw['frames_lost']==0,raw
            a=read_audio(d/'voice.wav');b=read_audio(test/'rx/rx_correct.wav')
            wrong=read_audio(test/'rx/rx_wrong.wav')
            correct=metric(a,b); incorrect=metric(a,wrong)
            print(cipher,op,'PCM frames=',raw['frames_ok'],'corr correct=',correct,'wrong=',incorrect,flush=True)
            assert correct>0.20, (cipher,op,'poor correct correlation',correct)
            if op!='invert': assert correct>incorrect+0.07, (cipher,op,correct,incorrect)
        test=d/'stft_signflip'
        if args.codec:
            if codec_opus(test/'tx/tx_wire.wav',test/'opus_20k.wav'):
                raw=run('rx',key,test/'opus_20k.wav',test/'opus_rx','stft','signflip')
                print('Opus 96k: frames_ok=%s frames_lost=%s acquisitions=%s corr=%s'%(raw['frames_ok'],raw['frames_lost'],raw['acquisitions'],metric(x,read_audio(test/'opus_rx/rx_correct.wav'))),flush=True)
        # Simulate late join after 80 full frames: modulo-64 is essential for mask correctness.
        with wave.open(str(test/'tx/tx_wire.wav'),'rb') as w:
            w.setpos(80*400);late_bytes=w.readframes(w.getnframes()-w.tell())
        with wave.open(str(test/'late.wav'),'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(20000);w.writeframes(late_bytes)
        stats=run('rx',key,test/'late.wav',test/'late_rx','stft','signflip')
        print('Late join: frames_ok=%s corr_to_source_tail=%s' % (stats['frames_ok'], metric(x[80*320:], read_audio(test/'late_rx/rx_correct.wav')[-120*320:])),flush=True)
        assert stats['frames_ok'] >= 95,stats
        # Direct API test of a plug-in: both TX and RX see frames and indexes.
        plug=d/'plug'; plug.mkdir()
        name='crypto.example_plugin:IdentityExample'
        run('tx',key,d/'voice.wav',plug/'tx','stft','signflip',plugin=name)
        summary=run('rx',key,plug/'tx/tx_wire.wav',plug/'rx','stft','signflip',plugin=name)
        assert summary['frames_ok']==200,summary
        print('PASS plugin, late join, six cipher variants, raw/correct/wrong WAV',flush=True)

if __name__=='__main__':main()
