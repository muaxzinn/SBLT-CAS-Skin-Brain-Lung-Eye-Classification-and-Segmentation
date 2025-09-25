const UI = {
    dropzone: null,fileInput: null,previewWrap: null,previewImg: null,previewName: null,
    btnChange: null,btnPredict: null,btnClear: null,loader: null,statusText: null,
    progress: null,progressBar: null,result: null,lowConfidence: null,
  };
  const STATE = {config:null, model:null, imageEl:null, warm:false, predicting:false, minConfidence:0.6};
  const $ = s=>document.querySelector(s); const clamp01=x=>Math.max(0,Math.min(1,x));
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const CLASS_NAMES_TH = {
    skin: 'ผิวหนัง',
    eye: 'ตา',
    lung: 'ปอด',
    brain: 'สมอง'
  };
  
  function setStatus(t,loading=false){UI.statusText.textContent=t;UI.loader.classList.toggle('hidden',!loading)}
  function setProgress(v){UI.progress.classList.remove('hidden');UI.progressBar.style.width=`${Math.round(clamp01(v)*100)}%`}
  function hideProgress(){UI.progress.classList.add('hidden');UI.progressBar.style.width='0%'}
  function enableControls(e){UI.btnPredict.disabled=!e;UI.btnClear.disabled=!e}
  
  function showPreview(file){
    const url=URL.createObjectURL(file); UI.previewImg.src=url; UI.previewName.textContent=file.name||'ภาพ';
    UI.previewWrap.classList.remove('hidden'); enableControls(true);
  }
  function clearPreview(){
    UI.previewImg.src = ''; UI.previewName.textContent = ''; UI.previewWrap.classList.add('hidden');
    UI.fileInput.value=''; UI.result.innerHTML='<p class="muted">จะแสดงเปอร์เซ็นต์หลังทำนาย</p>';
    UI.lowConfidence.classList.add('hidden'); enableControls(false);
  }
  
  async function loadConfig(){
    const res=await fetch('./config.json'); if(!res.ok) throw new Error(`โหลด config.json ล้มเหลว ${res.status}`);
    const cfg=await res.json(); if(!cfg.router||!cfg.router.path||!cfg.router.labels||!cfg.router.redirect) throw new Error('config.router ไม่ครบ');
    STATE.config=cfg; if(typeof cfg.router.minConfidence==='number') STATE.minConfidence=clamp01(cfg.router.minConfidence);
  }
  
  async function loadRouterModel(){
    setStatus('กำลังโหลด Router Model…', true); setProgress(0.1);
    STATE.model=await tf.loadLayersModel(STATE.config.router.path); setProgress(0.5);
    tf.tidy(()=>{const warm=tf.zeros([1,224,224,3]); const out=STATE.model.predict(warm); (Array.isArray(out)?out:[out]).forEach(t=>t.dataSync())});
    setProgress(0.9); setStatus('พร้อมใช้งาน ✅'); hideProgress(); STATE.warm=true;
  }
  
  function preprocess(imgEl){
    return tf.tidy(()=> {
      let t=tf.browser.fromPixels(imgEl); if(t.shape[2]===4) t=t.slice([0,0,0],[-1,-1,3]);
      t=tf.image.resizeBilinear(t,[224,224]).toFloat().div(255).expandDims(0);
      return t;
    });
  }
  
  function renderProbBars(labels, probs){
    UI.result.innerHTML = labels.map((lb,i)=>{
      const pct=Math.round(probs[i]*100);
      const nameTh = CLASS_NAMES_TH[lb] || lb;
      return `<div class="kv"><div class="label">${i+1}. ${nameTh}</div><div class="bar"><span style="width:${pct}%"></span></div></div>`;
    }).join('');
  }
  
  async function predictAndRoute(){
    if(STATE.predicting) return;
    if(!STATE.model||!STATE.warm){setStatus('โมเดลยังไม่พร้อม…', true);return}
    if(!STATE.imageEl){setStatus('โปรดเลือกภาพก่อน');return}
    STATE.predicting=true; setStatus('กำลังทำนาย…',true); setProgress(0.15);
    try{
      const {labels,redirect}=STATE.config.router;
      const input=preprocess(STATE.imageEl); setProgress(0.35);
      const logits=STATE.model.predict(input);
      const probsT=tf.tidy(()=>tf.softmax(logits));
      const probs=Array.from(await probsT.data()); input.dispose(); probsT.dispose(); logits.dispose?.();
      setProgress(0.65); renderProbBars(labels,probs);
      let idx=0, val=-1; probs.forEach((p,i)=>{if(p>val){val=p;idx=i}});
      const top = labels[idx], conf = val;
      if(conf>=STATE.minConfidence){
        const nameTh = CLASS_NAMES_TH[top] || top; // แปลงชื่อคลาสเป็นภาษาไทย
        setStatus(`ผลลัพธ์: ${nameTh} (${(conf*100).toFixed(1)}%) → ไปยังหน้าโมเดล…`, true); setProgress(0.9);
        const target=redirect[top]; if(!target) throw new Error(`ไม่พบ redirect ของ "${top}"`);
        await sleep(600); window.location.href=target;
      }else{
        const nameTh = CLASS_NAMES_TH[top] || top; // แปลงชื่อคลาสเป็นภาษาไทย
        setStatus(`ความมั่นใจต่ำ (ผลลัพธ์สูงสุด: ${nameTh}, ${(conf*100).toFixed(1)}%) · เลือกปลายทางด้านล่าง`, false);
        UI.lowConfidence.classList.remove('hidden'); hideProgress();
      }
    }catch(err){console.error(err); setStatus(`ผิดพลาด: ${err.message}`); hideProgress()}
    finally{STATE.predicting=false}
  }
  
  function wireDragAndDrop(){
    const dz=UI.dropzone, fi=UI.fileInput, openPicker=()=>fi.click();
    dz.addEventListener('click',openPicker);
    dz.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openPicker()}});
    dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover')});
    dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
    dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('dragover');const f=e.dataTransfer.files?.[0]; if(f) handleFile(f)});
    fi.addEventListener('change',e=>{const f=e.target.files?.[0]; if(f) handleFile(f)});
    window.addEventListener('paste',e=>{const it=[...(e.clipboardData?.items||[])].find(x=>x.type.startsWith('image/')); if(it){const f=it.getAsFile(); if(f) handleFile(f)}});
    UI.btnChange.addEventListener('click',openPicker);
  }
  function handleFile(file){
    if(!file.type.startsWith('image/')){setStatus('ไฟล์ไม่ใช่รูปภาพ');return}
    showPreview(file); const img=new Image();
    img.onload=()=>{STATE.imageEl=img; setStatus('ภาพพร้อมทำนาย')};
    img.onerror=()=>setStatus('โหลดภาพไม่สำเร็จ');
    img.src=URL.createObjectURL(file);
  }
  function wireActions(){
    UI.btnPredict.addEventListener('click', predictAndRoute);
    UI.btnClear.addEventListener('click', ()=>{clearPreview(); STATE.imageEl=null; setStatus('พร้อมใช้งาน ✅')});
  }
  
  async function main(){
    UI.dropzone = $('#dropzone'); UI.fileInput = $('#fileInput'); UI.previewWrap = $('#previewWrap');
    UI.previewImg = $('#previewImg'); UI.previewName = $('#previewName');
    UI.btnChange = $('#btnChange'); UI.btnPredict = $('#btnPredict'); UI.btnClear = $('#btnClear');
    UI.loader = $('#loader'); UI.statusText = $('#statusText');
    UI.progress = $('#progress'); UI.progressBar = $('#progressBar');
    UI.result = $('#result'); UI.lowConfidence = $('#lowConfidence');
    setStatus('กำลังเริ่มระบบ…', true); enableControls(false);
    wireDragAndDrop(); wireActions();
    try{ await loadConfig(); await loadRouterModel(); setStatus('พร้อมใช้งาน ✅') } 
    catch(e){ console.error(e); setStatus(`โหลดไม่สำเร็จ: ${e.message}`) }
  }
  document.addEventListener('DOMContentLoaded', main);
  