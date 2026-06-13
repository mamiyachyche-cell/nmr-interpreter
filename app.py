import os
import zipfile
import tempfile
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import nmrglue as ng

app = Flask(__name__)
CORS(app, origins="*")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML = open('index.html').read() if os.path.exists('index.html') else '<h1>index.html not found</h1>'

def find_bruker_dir(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'fid' in files or '1r' in files:
            return root
    return None

def pick_peaks(ppm_scale, spectrum, threshold_factor=0.05):
    peaks = []
    noise = np.std(spectrum[:100])
    threshold = max(noise * 10, np.max(spectrum) * threshold_factor)
    for i in range(2, len(spectrum) - 2):
        if (spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1] and
            spectrum[i] > spectrum[i-2] and spectrum[i] > spectrum[i+2] and
            spectrum[i] > threshold):
            peaks.append({'ppm': round(float(ppm_scale[i]), 3), 'intensity': round(float(spectrum[i]), 1)})
    peaks.sort(key=lambda x: x['ppm'], reverse=True)
    return merge_close_peaks(peaks)

def merge_close_peaks(peaks, min_separation=0.02):
    if not peaks: return peaks
    merged = [peaks[0]]
    for p in peaks[1:]:
        if abs(p['ppm'] - merged[-1]['ppm']) < min_separation:
            if p['intensity'] > merged[-1]['intensity']: merged[-1] = p
        else: merged.append(p)
    return merged

def estimate_multiplicity(ppm, ppm_scale, spectrum, window=0.08):
    idx = np.argmin(np.abs(ppm_scale - ppm))
    half = int(window / abs(ppm_scale[1] - ppm_scale[0]))
    region = spectrum[max(0, idx-half):idx+half]
    if len(region) == 0: return 's'
    noise = np.std(spectrum[:100])
    threshold = noise * 5
    sub, in_peak = 0, False
    for v in region:
        if v > threshold and not in_peak: sub += 1; in_peak = True
        elif v <= threshold: in_peak = False
    if sub <= 1: return 's'
    elif sub == 2: return 'd'
    elif sub == 3: return 't'
    elif sub == 4: return 'q'
    else: return 'm'

def process_bruker_zip(zip_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)
        bruker_dir = find_bruker_dir(tmpdir)
        if not bruker_dir:
            return None, 'Could not find Bruker data folder inside zip.'
        try:
            pdata_path = os.path.join(bruker_dir, 'pdata', '1')
            if os.path.exists(pdata_path):
                dic, data = ng.bruker.read_pdata(pdata_path)
            else:
                dic, data = ng.bruker.read(bruker_dir)
                data = ng.bruker.remove_digital_filter(dic, data)
                data = ng.proc_base.zf_size(data, 32768)
                data = ng.proc_base.fft(data)
                data = ng.proc_base.ps(data, p0=0, p1=0)
            udic = ng.bruker.guess_udic(dic, data)
            uc = ng.fileiobase.uc_from_udic(udic)
            ppm_scale = uc.ppm_scale()
            spectrum = data.real
            spectrum = spectrum - np.min(spectrum)
            if np.max(spectrum) > 0: spectrum = spectrum / np.max(spectrum) * 100
            mask = (ppm_scale >= 0) & (ppm_scale <= 15)
            ppm_r, spec_r = ppm_scale[mask], spectrum[mask]
            peaks = pick_peaks(ppm_r, spec_r)
            for p in peaks:
                p['multiplicity'] = estimate_multiplicity(p['ppm'], ppm_r, spec_r)
            meta = {}
            if 'acqus' in dic:
                acq = dic['acqus']
                meta['frequency'] = round(float(acq.get('SFO1', 400)), 1)
                sc = acq.get('SOLVENT', 'DMSO')
                sm = {'DMSO':'DMSO-d6','CDCl3':'CDCl3','D2O':'D2O','MeOD':'MeOD','CD3OD':'MeOD','Acetone':'Acetone-d6'}
                meta['solvent'] = sm.get(sc, sc)
            return {'peaks': peaks, 'meta': meta}, None
        except Exception as e:
            return None, f'Error reading NMR data: {str(e)}'

@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Please upload a .zip file'}), 400
    save_path = os.path.join(UPLOAD_FOLDER, 'upload.zip')
    f.save(save_path)
    result, error = process_bruker_zip(save_path)
    if error: return jsonify({'error': error}), 400
    return jsonify({'success': True, 'peaks': result['peaks'], 'meta': result['meta'], 'peak_count': len(result['peaks'])})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port)
