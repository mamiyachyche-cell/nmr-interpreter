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

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NMR Interpreter</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f0;color:#1a1a1a;min-height:100vh}
.topbar{background:#fff;border-bottom:1px solid #e5e5e0;padding:0 2rem;height:56px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}
.topbar h1{font-size:16px;font-weight:600}
.badge{font-size:11px;background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:20px;font-weight:500;margin-left:8px}
.container{max-width:780px;margin:0 auto;padding:2rem 1rem}
.card{background:#fff;border:1px solid #e5e5e0;border-radius:12px;padding:1.5rem;margin-bottom:1rem}
.card-title{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#888;margin-bottom:1rem}
label{font-size:12px;color:#666;display:block;margin-bottom:5px;font-weight:500}
input[type=text],textarea,select{width:100%;padding:9px 12px;border:1px solid #ddd;border-radius:8px;font-size:14px;font-family:inherit;background:#fff;color:#1a1a1a;transition:border-color .15s}
input[type=text]:focus,textarea:focus,select:focus{outline:none;border-color:#4a9eff;box-shadow:0 0 0 3px rgba(74,158,255,.1)}
textarea{resize:vertical;line-height:1.6}
.mode-bar{display:flex;background:#f0f0eb;border-radius:10px;padding:4px;gap:4px;margin-bottom:1.5rem}
.mode-btn{flex:1;padding:9px 8px;border:none;border-radius:8px;background:transparent;cursor:pointer;font-size:13px;color:#666;font-family:inherit;transition:all .15s}
.mode-btn.active{background:#fff;color:#1a1a1a;font-weight:600;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.drop-zone{border:2px dashed #ddd;border-radius:10px;padding:3rem 1rem;text-align:center;cursor:pointer;transition:all .15s;background:#fafaf8}
.drop-zone:hover,.drop-zone.drag{border-color:#4a9eff;background:#f0f7ff}
.drop-zone p{font-size:15px;color:#555;margin-bottom:4px;font-weight:500}
.drop-zone small{font-size:12px;color:#999}
.peak-row{display:grid;grid-template-columns:1fr 100px 80px 80px 32px;gap:8px;margin-bottom:8px;align-items:center}
.peak-header{display:grid;grid-template-columns:1fr 100px 80px 80px 32px;gap:8px;margin-bottom:6px}
.peak-header span{font-size:11px;color:#999;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.rm-btn{width:32px;height:32px;border-radius:8px;border:1px solid #eee;background:#fff;cursor:pointer;color:#ccc;font-size:18px;display:flex;align-items:center;justify-content:center;transition:all .15s}
.rm-btn:hover{border-color:#ff4444;color:#ff4444}
.add-btn{font-size:13px;color:#4a9eff;background:none;border:none;cursor:pointer;padding:6px 0;display:flex;align-items:center;gap:5px;margin-top:4px;font-family:inherit;font-weight:500}
.analyze-btn{width:100%;padding:14px;border-radius:10px;border:none;background:#1a1a1a;color:#fff;font-size:15px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;margin-top:1rem;font-family:inherit;transition:background .15s}
.analyze-btn:hover{background:#333}
.eg-btn{font-size:12px;color:#4a9eff;background:#f0f7ff;border:none;border-radius:6px;padding:5px 12px;cursor:pointer;margin-top:8px;font-family:inherit;font-weight:500}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.tip{font-size:12px;color:#999;margin-top:6px;line-height:1.5}
.status{padding:10px 14px;border-radius:8px;font-size:13px;margin-top:10px;display:none;line-height:1.6}
.status.ok{background:#e8f5e9;color:#2e7d32;display:block}
.status.err{background:#ffebee;color:#c62828;display:block}
.status.info{background:#e3f2fd;color:#1565c0;display:block}
.peak-pill{display:inline-flex;align-items:center;gap:5px;background:#f5f5f0;border:1px solid #e5e5e0;border-radius:6px;padding:4px 10px;font-size:13px;margin:3px}
.peak-pill .mult{font-size:11px;color:#999}
.copy-box{background:#f5f5f0;border:1px solid #e5e5e0;border-radius:10px;padding:1.25rem;margin-top:1rem}
.copy-box h3{font-size:13px;font-weight:600;color:#555;margin-bottom:10px}
.copy-box pre{font-size:13px;font-family:monospace;white-space:pre-wrap;color:#333;background:#fff;border:1px solid #e5e5e0;border-radius:8px;padding:12px;max-height:220px;overflow-y:auto;margin-bottom:12px;line-height:1.6}
.copy-btn{padding:10px 20px;border-radius:8px;border:none;background:#4a9eff;color:#fff;font-size:14px;font-weight:600;cursor:pointer;font-family:inherit;transition:background .15s}
.copy-btn:hover{background:#2979ff}
.copy-btn.copied{background:#2e7d32}
.step-box{background:#fff8e1;border:1px solid #ffe082;border-radius:10px;padding:1.25rem;margin-top:1rem;font-size:14px;color:#5d4037;line-height:1.8}
</style>
</head>
<body>
<div class="topbar">
  <h1>&#9879;&#65039; NMR Interpreter <span class="badge">Free &amp; Open Access</span></h1>
  <span style="font-size:12px;color:#999;">Powered by Claude AI &middot; Developed by Mamiya Chowdhury, Tennessee Tech University</span>
</div>
<div class="container">
  <div class="card" style="border-color:#c8e6c9;background:#f1f8e9;">
    <div class="card-title" style="color:#2e7d32;">&#128161; How this works &mdash; completely free</div>
    <p style="font-size:14px;color:#388e3c;line-height:1.8;">
      1. Upload your Bruker zip file OR type your peaks manually<br>
      2. Click Analyze &mdash; it builds a prompt and copies it to your clipboard<br>
      3. Paste into Claude chat at <a href="https://claude.ai" target="_blank" style="color:#1565c0;">claude.ai</a> &rarr; get full AI analysis instantly!
    </p>
  </div>
  <div class="mode-bar">
    <button class="mode-btn active" id="mb-upload" onclick="setMode('upload')">&#128194; Upload Bruker file</button>
    <button class="mode-btn" id="mb-peaks" onclick="setMode('peaks')">&#128221; Type peaks manually</button>
    <button class="mode-btn" id="mb-verify" onclick="setMode('verify')">&#9989; Verify match</button>
  </div>
  <div id="panel-upload">
    <div class="card">
      <div class="card-title">&#128203; Prepare your Bruker folder</div>
      <p style="font-size:14px;color:#555;line-height:1.8;">
        1. Find your NMR data folder &mdash; it contains files called <code style="background:#f5f5f0;padding:1px 6px;border-radius:4px;font-size:13px">fid</code> and <code style="background:#f5f5f0;padding:1px 6px;border-radius:4px;font-size:13px">acqus</code><br>
        2. Right-click the folder &rarr; <strong>Send to &rarr; Compressed (zipped) folder</strong><br>
        3. Upload that zip file below &#8595;
      </p>
    </div>
    <div class="card">
      <div class="card-title">&#128228; Upload your Bruker zip file</div>
      <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()" ondragover="dragOver(event)" ondragleave="dragLeave(event)" ondrop="dropFile(event)">
        <p style="font-size:2.5rem;margin-bottom:8px;">&#128193;</p>
        <p>Click to select or drag your zip file here</p>
        <small>Your Bruker folder zipped &mdash; e.g. compound1.zip</small>
      </div>
      <input type="file" id="file-input" accept=".zip" style="display:none" onchange="fileSelected(event)">
      <div id="upload-status" class="status"></div>
    </div>
    <div class="card" id="peaks-preview" style="display:none">
      <div class="card-title">&#9989; Peaks extracted from your file</div>
      <div id="extracted-meta" style="font-size:13px;color:#666;margin-bottom:12px;"></div>
      <div id="extracted-peaks"></div>
      <div style="margin-top:16px;">
        <label>Your compound name / description</label>
        <input type="text" id="u-compound" placeholder="e.g. Furfural-derived NAH from furfural + benzohydrazide">
      </div>
    </div>
  </div>
  <div id="panel-peaks" style="display:none">
    <div class="card">
      <div class="card-title">&#129514; Compound description</div>
      <textarea id="p-comp" rows="2" placeholder="e.g. Furfural-derived N-acyl hydrazone from furfural + benzohydrazide, DMSO-d6, 400 MHz"></textarea>
      <button class="eg-btn" onclick="loadEg()">&#10024; Load NAH example</button>
    </div>
    <div class="card">
      <div class="card-title">&#128202; &sup1;H NMR peaks</div>
      <div class="peak-header">
        <span>&#948; (ppm)</span><span>Multiplicity</span><span>Integration</span><span>J (Hz)</span><span></span>
      </div>
      <div id="pklist"></div>
      <button class="add-btn" onclick="addRow()">+ Add peak</button>
      <p class="tip">s=singlet &middot; d=doublet &middot; t=triplet &middot; dd=doublet of doublets &middot; m=multiplet &middot; br s=broad singlet</p>
    </div>
    <div class="card">
      <div class="two-col">
        <div><label>Solvent</label>
          <select id="p-solv"><option>DMSO-d6</option><option>CDCl3</option><option>D2O</option><option>MeOD</option><option>Acetone-d6</option></select>
        </div>
        <div><label>Frequency (MHz)</label>
          <select id="p-freq"><option>400</option><option>300</option><option>500</option><option>600</option></select>
        </div>
      </div>
    </div>
  </div>
  <div id="panel-verify" style="display:none">
    <div class="card">
      <div class="card-title">&#128300; Expected structure</div>
      <textarea id="v-str" rows="2" placeholder="e.g. (E)-N'-(furan-2-ylmethylene)benzohydrazide"></textarea>
    </div>
    <div class="card">
      <div class="card-title">&#128202; Your actual &sup1;H NMR peaks</div>
      <textarea id="v-pks" rows="7" style="font-family:monospace;font-size:13px" placeholder="&#948; 11.42 (br s, 1H)&#10;&#948; 8.31 (s, 1H)&#10;&#948; 7.85 (d, J=7.5 Hz, 2H)"></textarea>
      <div style="margin-top:10px;">
        <label>Solvent</label>
        <select id="v-solv"><option>DMSO-d6</option><option>CDCl3</option><option>D2O</option><option>MeOD</option></select>
      </div>
    </div>
  </div>
  <button class="analyze-btn" onclick="analyze()">&#10024; Build prompt &amp; copy to clipboard</button>
  <div id="prompt-area" style="display:none">
    <div class="copy-box">
      <h3>&#128203; Your analysis prompt &mdash; ready to paste into Claude</h3>
      <pre id="prompt-text"></pre>
      <button class="copy-btn" id="copy-btn" onclick="copyPrompt()">&#128203; Copy to clipboard</button>
    </div>
    <div class="step-box">
      <strong>Next step:</strong> Click "Copy to clipboard" above, go to <a href="https://claude.ai" target="_blank" style="color:#1565c0;font-weight:600;">claude.ai</a>, paste and press Enter &mdash; Claude gives you the full analysis!
    </div>
  </div>
</div>
<script>
let mode='upload',rc=0,extractedPeaks=[],extractedMeta={};
function setMode(m){mode=m;['upload','peaks','verify'].forEach(p=>{document.getElementById('panel-'+p).style.display=p===m?'block':'none';document.getElementById('mb-'+p).classList.toggle('active',p===m);});}
function dragOver(e){e.preventDefault();document.getElementById('drop-zone').classList.add('drag');}
function dragLeave(){document.getElementById('drop-zone').classList.remove('drag');}
function dropFile(e){e.preventDefault();dragLeave();const f=e.dataTransfer.files[0];if(f)uploadFile(f);}
function fileSelected(e){const f=e.target.files[0];if(f)uploadFile(f);}
async function uploadFile(file){
  const statusEl=document.getElementById('upload-status');
  const previewEl=document.getElementById('peaks-preview');
  previewEl.style.display='none';
  if(!file.name.endsWith('.zip')){statusEl.className='status err';statusEl.textContent='Please upload a .zip file';return;}
  statusEl.className='status info';statusEl.textContent='Reading your Bruker file...';
  const formData=new FormData();formData.append('file',file);
  try{
    const res=await fetch('/upload',{method:'POST',body:formData});
    const data=await res.json();
    if(data.error){statusEl.className='status err';statusEl.textContent='Error: '+data.error;return;}
    extractedPeaks=data.peaks;extractedMeta=data.meta;
    statusEl.className='status ok';statusEl.textContent='Found '+data.peak_count+' peaks in your spectrum!';
    document.getElementById('extracted-meta').innerHTML='Solvent: <strong>'+(data.meta.solvent||'?')+'</strong> &middot; Frequency: <strong>'+(data.meta.frequency||'?')+' MHz</strong> &middot; '+data.peak_count+' peaks';
    document.getElementById('extracted-peaks').innerHTML=data.peaks.map(p=>'<span class="peak-pill">&#948; '+p.ppm+' <span class="mult">'+p.multiplicity+'</span></span>').join('');
    previewEl.style.display='block';
  }catch(e){statusEl.className='status err';statusEl.textContent='Upload failed: '+e.message;}
}
function addRow(ppm,mult,intg,j){
  rc++;const d=document.createElement('div');d.className='peak-row';d.id='r'+rc;
  d.innerHTML='<input type="text" value="'+(ppm||'')+'" placeholder="e.g. 8.31">'
    +'<select><option value="">&#8212;</option>'+['s','br s','d','t','q','dd','dt','ddd','m'].map(x=>'<option'+(mult===x?' selected':'')+'>'+x+'</option>').join('')+'</select>'
    +'<input type="text" value="'+(intg||'')+'" placeholder="1H">'
    +'<input type="text" value="'+(j||'')+'" placeholder="7.5">'
    +'<button class="rm-btn" onclick="document.getElementById(\'r'+rc+'\').remove()">&times;</button>';
  document.getElementById('pklist').appendChild(d);
}
function loadEg(){
  document.getElementById('p-comp').value="Furfural-derived NAH from furfural + benzohydrazide. Expected: (E)-N'-(furan-2-ylmethylene)benzohydrazide. DMSO-d6, 400 MHz.";
  document.getElementById('pklist').innerHTML='';rc=0;
  [['11.42','br s','1H',''],['8.31','s','1H',''],['7.85','d','2H','7.5'],['7.55','t','1H','7.5'],['7.46','t','2H','7.5'],['7.70','s','1H',''],['6.62','m','1H','']].forEach(p=>addRow(...p));
}
function buildPrompt(){
  if(mode==='upload'){
    if(!extractedPeaks.length){alert('Please upload your Bruker zip file first!');return null;}
    const comp=document.getElementById('u-compound').value||'Furfural-derived N-acyl hydrazone';
    const peakList=extractedPeaks.map(p=>'delta '+p.ppm+' ('+p.multiplicity+')').join('\\n');
    return '[NMR INTERPRETER - BRUKER FILE UPLOAD MODE]\\nCompound: '+comp+'\\nSolvent: '+(extractedMeta.solvent||'DMSO-d6')+' | Frequency: '+(extractedMeta.frequency||400)+' MHz\\n\\nAutomatically extracted peaks:\\n'+peakList+'\\n\\nPlease assign each peak to a specific proton and functional group, explain WHY it appears at that chemical shift, comment on multiplicity, discuss E/Z isomerism if NH or CH=N peaks are present, and give a final verdict. Use plain educational language for a masters student learning NMR.';
  }
  if(mode==='peaks'){
    const comp=document.getElementById('p-comp').value||'Not specified';
    let peaks='';
    document.querySelectorAll('#pklist .peak-row').forEach(row=>{
      const ins=row.querySelectorAll('input,select');
      if(ins[0].value)peaks+='delta '+ins[0].value+' ('+(ins[1].value||'?')+', '+(ins[2].value||'?H')+(ins[3].value?', J='+ins[3].value+' Hz':'')+')'+'\\n';
    });
    if(!peaks.trim()){alert('Please add your peaks first!');return null;}
    return '[NMR INTERPRETER - MANUAL PEAKS MODE]\\nCompound: '+comp+'\\nSolvent: '+document.getElementById('p-solv').value+' | Frequency: '+document.getElementById('p-freq').value+' MHz\\n\\nPeaks:\\n'+peaks+'\\nPlease assign each peak, explain the chemistry, comment on multiplicities, discuss E/Z isomerism if relevant, and give a verdict. Use plain educational language.';
  }
  if(mode==='verify'){
    const str=document.getElementById('v-str').value,pks=document.getElementById('v-pks').value;
    if(!str.trim()||!pks.trim()){alert('Please fill in both fields!');return null;}
    return '[NMR INTERPRETER - VERIFY MATCH MODE]\\nExpected structure: '+str+'\\nSolvent: '+document.getElementById('v-solv').value+'\\n\\nActual peaks:\\n'+pks+'\\n\\nPlease assign each peak, check multiplicities and integrations, flag unexpected or missing peaks, and give a clear verdict. Use educational language.';
  }
}
function analyze(){const prompt=buildPrompt();if(!prompt)return;document.getElementById('prompt-text').textContent=prompt;document.getElementById('prompt-area').style.display='block';document.getElementById('prompt-area').scrollIntoView({behavior:'smooth'});}
function copyPrompt(){navigator.clipboard.writeText(document.getElementById('prompt-text').textContent).then(()=>{const btn=document.getElementById('copy-btn');btn.textContent='Copied!';btn.classList.add('copied');setTimeout(()=>{btn.textContent='Copy to clipboard';btn.classList.remove('copied');},2000);});}
addRow();addRow();addRow();
</script>
</body>
</html>"""

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
