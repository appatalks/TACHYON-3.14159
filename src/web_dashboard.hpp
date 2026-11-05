#pragma once
// TACHYON-π embedded dashboard (single-file, zero-dependency).
static const char* DASHBOARD_HTML = R"====(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TACHYON-&pi; &mdash; heterogeneous CPU+GPU &pi; engine</title>
<style>
:root{
  --bg:#0a0e17;--panel:#111726;--panel2:#0d1320;--line:#1e2a42;
  --txt:#dbe4f5;--dim:#8b98b3;--acc:#4f8cff;--acc2:#8f5bff;
  --cpu:#4f8cff;--gpu:#ff9440;--ok:#2fd57b;--bad:#ff5470;--warn:#ffc857;
  --mono:ui-monospace,'JetBrains Mono','Fira Code',Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 700px at 75% -10%,#14203a 0%,var(--bg) 55%);
 color:var(--txt);font:14px/1.5 system-ui,'Segoe UI',Roboto,sans-serif;min-height:100vh}
header{display:flex;align-items:center;gap:14px;padding:12px 22px;border-bottom:1px solid var(--line);
 position:sticky;top:0;background:rgba(10,14,23,.88);backdrop-filter:blur(6px);z-index:10}
.logo{font-family:var(--mono);font-size:19px;font-weight:700;letter-spacing:.06em}
.logo b{background:linear-gradient(90deg,var(--acc),var(--acc2));-webkit-background-clip:text;
 background-clip:text;color:transparent}
.tag{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:20px;padding:2px 10px}
.spacer{flex:1}
.dot{width:9px;height:9px;border-radius:50%;background:#456;display:inline-block;margin-right:6px}
.dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}
.dot.off{background:var(--bad);box-shadow:0 0 8px var(--bad)}
nav{display:flex;gap:4px}
nav button{background:none;border:1px solid transparent;color:var(--dim);padding:6px 14px;
 border-radius:8px;cursor:pointer;font-size:13.5px}
nav button:hover{color:var(--txt)}
nav button.act{color:var(--txt);background:var(--panel);border-color:var(--line)}
main{padding:20px 22px;max-width:1260px;margin:0 auto}
section{display:none}section.act{display:block}
.card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
 border-radius:14px;padding:18px;margin-bottom:18px}
h2{margin:0 0 12px;font-size:15px;font-weight:600}
h2 small{color:var(--dim);font-weight:400;margin-left:8px}
label{display:block;font-size:11px;color:var(--dim);margin:0 0 4px;text-transform:uppercase;letter-spacing:.08em}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end}
input[type=number],input[type=text],select{background:#0b1120;border:1px solid var(--line);
 color:var(--txt);border-radius:8px;padding:8px 10px;font-family:var(--mono);font-size:14px;width:140px}
select{width:auto}
input:focus,select:focus{outline:none;border-color:var(--acc)}
.chk{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:13px;padding-bottom:8px;cursor:pointer}
.chk input{accent-color:var(--acc);width:15px;height:15px}
.btn{background:linear-gradient(90deg,var(--acc),var(--acc2));border:none;color:#fff;font-weight:600;
 padding:9px 22px;border-radius:9px;cursor:pointer;font-size:14px}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn.ghost{background:var(--panel);border:1px solid var(--line);color:var(--txt);font-weight:500}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:950px){.grid2{grid-template-columns:1fr}}
.pbar{height:10px;border-radius:6px;background:#0b1120;border:1px solid var(--line);overflow:hidden;margin:6px 0}
.pbar i{display:block;height:100%;border-radius:6px;background:linear-gradient(90deg,var(--acc),var(--acc2));
 transition:width .25s}
.mono{font-family:var(--mono)} .k{color:var(--dim)} .v{font-family:var(--mono)}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px 18px;margin-top:4px}
.big{font-size:30px;font-weight:700;font-family:var(--mono)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.phase{font-family:var(--mono);color:var(--acc);font-size:13px}
.pre{font-family:var(--mono);font-size:12.5px;background:#0b1120;border:1px solid var(--line);
 border-radius:8px;padding:10px 12px;overflow-x:auto;white-space:nowrap;color:#a8c6ff;word-break:break-all}
table{border-collapse:collapse;width:100%;font-size:13px}
th{color:var(--dim);text-align:left;font-weight:500;padding:6px 10px;border-bottom:1px solid var(--line);
 font-size:11px;text-transform:uppercase;letter-spacing:.06em}
td{padding:6px 10px;border-bottom:1px solid #16203255;font-family:var(--mono);font-size:12.5px}
tr:hover td{background:#141d31}
.legend{display:flex;gap:18px;font-size:12.5px;color:var(--dim);margin:8px 0}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
#toast{position:fixed;right:20px;bottom:20px;display:flex;flex-direction:column;gap:8px;z-index:50}
.tst{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);
 padding:10px 16px;border-radius:10px;font-size:13px;box-shadow:0 6px 24px #0008}
.tst.bad{border-left-color:var(--bad)}
.note{font-size:12px;color:var(--dim);margin-top:8px;line-height:1.45}
svg text{font-family:var(--mono)}
.jobs{max-height:320px;overflow:auto}
.jobs tr{cursor:pointer}
.jobs tr.sel td{background:#182742}
.small{font-size:12px}
</style>
</head>
<body>
<header>
 <div class="logo"><b>TACHYON-&pi;</b></div>
 <div class="tag">CPU + GPU arbitrary-precision &pi;</div>
 <div class="spacer"></div>
 <div class="small"><span class="dot" id="hdot"></span><span id="htxt" class="k">checking&hellip;</span></div>
 <nav>
  <button data-tab="run" class="act">Run</button>
  <button data-tab="map">Device mapping</button>
  <button data-tab="digits">Digits</button>
  <button data-tab="bench">Benchmarks</button>
 </nav>
</header>
<main>

<section id="tab-run" class="act">
 <div class="card">
  <h2>New computation <small>Chudnovsky binary splitting &rarr; exact integer &lfloor;&pi;&middot;10<sup>D</sup>&rfloor;</small></h2>
  <div class="row">
   <div><label>Preset</label>
    <select id="dpresets" onchange="if(this.value)$('digits').value=this.value">
     <option value="10000">10&nbsp;000</option><option value="100000">100&nbsp;000</option>
     <option value="1000000" selected>1&nbsp;million</option><option value="10000000">10&nbsp;million</option>
     <option value="100000000">100&nbsp;million</option><option value="">custom&hellip;</option>
    </select></div>
   <div><label>Digits</label><input type="number" id="digits" value="1000000" min="2" max="1000000000"></div>
   <div><label>Backend</label><select id="backend">
    <option value="cpu">cpu (GMP)</option><option value="gpu">gpu (Stage&nbsp;3)</option>
    <option value="hybrid">hybrid (Stage&nbsp;4)</option></select></div>
   <div><label>Leaf</label><input type="number" id="leaf" value="8" min="1" max="1024" style="width:84px"></div>
   <div><label>Guard</label><input type="number" id="guard" value="30" min="10" style="width:84px"></div>
   <label class="chk"><input type="checkbox" id="verify" checked>BBP verify</label>
   <button class="btn" id="gorun" onclick="runJob()">Compute &pi;</button>
  </div>
 </div>
 <div class="grid2">
  <div class="card">
   <h2>Live progress <small>server-sent events</small></h2>
   <div id="live"><div class="k">No job running.</div></div>
   <div class="pbar" style="margin-top:14px"><i id="pbig" style="width:0"></i></div>
   <div class="phase" id="ptxt">&mdash;</div>
  </div>
  <div class="card">
   <h2>Result <small id="rid" class="mono"></small></h2>
   <div id="result"><div class="k">Run a computation to see the result.</div></div>
  </div>
 </div>
 <div class="card">
  <h2>Job queue <small>this server session &middot; click a row to load it</small></h2>
  <div class="jobs"><table id="jobs"><tr><th>id</th><th>digits</th><th>backend</th><th>state</th>
   <th>phase</th><th>total s</th><th>digits/s</th><th>RSS MB</th><th>verify</th></tr></table></div>
 </div>
</section>

<section id="tab-map">
 <div class="card">
  <h2>Hardware mapping <small>where each binary-splitting level executes</small></h2>
  <div class="row">
   <div><label>Digits</label><input type="number" id="mdigits" value="1000000"></div>
   <div><label>Backend</label><select id="mbackend"><option>cpu</option><option>gpu</option><option>hybrid</option></select></div>
   <button class="btn ghost" onclick="drawMap()">Recompute plan</button>
   <div class="spacer"></div>
   <div class="legend"><span><i style="background:var(--cpu)"></i>CPU (GMP)</span>
    <span><i style="background:var(--gpu)"></i>GPU (CUDA NTT, model)</span></div>
  </div>
  <div id="mapinfo" class="note"></div>
  <svg id="dag" width="1180" height="540" style="width:100%;height:auto;margin-top:10px"></svg>
  <div class="note" id="mapnote"></div>
 </div>
 <div class="card">
  <h2>Per-level cost table</h2>
  <div style="max-height:420px;overflow:auto"><table id="maptbl"><tr><th>level</th><th>terms / node</th>
   <th>nodes</th><th>node bits</th><th>device</th><th>cpu ms</th><th>gpu ms</th><th>xfer ms</th></tr></table></div>
 </div>
</section>

<section id="tab-digits">
 <div class="card">
  <h2>Digit analytics <small>completed jobs with &le;10M digits kept in server memory</small></h2>
  <div class="row">
   <div><label>Job</label><select id="djob" onchange="loadStats()"></select></div>
   <div><label>Find pattern</label><input type="text" id="dpattern" value="0123456789" style="width:220px"></div>
   <button class="btn ghost" onclick="findPat()">Search</button>
   <div class="spacer"></div>
   <div id="findres" class="small mono k"></div>
  </div>
  <svg id="freq" width="1140" height="250" style="width:100%;height:auto;margin-top:14px"></svg>
  <div id="statinfo" class="note"></div>
 </div>
 <div class="card">
  <h2>Digit browser</h2>
  <div class="row">
   <div><label>Offset (after &ldquo;3.&rdquo;)</label><input type="number" id="doff" value="0" min="0" style="width:170px"></div>
   <button class="btn ghost" onclick="loadWindow()">Show window</button>
   <div class="note" style="padding-bottom:8px">1000 digits per line</div>
  </div>
  <div class="pre" id="dwin" style="margin-top:10px;white-space:pre-wrap">&mdash;</div>
 </div>
</section>

<section id="tab-bench">
 <div class="card">
  <h2>End-to-end timings <small>log&ndash;log &middot; every recorded run &middot; y-cruncher reference (same machine)</small></h2>
  <div class="legend"><span><i style="background:#4f8cff"></i>TACHYON cpu</span>
   <span><i style="background:#ff9440"></i>TACHYON gpu / hybrid</span>
   <span><i style="background:#2fd57b"></i>y-cruncher (reference bar)</span></div>
  <svg id="bchart" width="1140" height="430" style="width:100%;height:auto"></svg>
  <div class="note">CLI records (benchmarks/results.jsonl) are single-thread baseline runs; server records are dashboard jobs
   (include BBP verify when enabled). y-cruncher bars from BENCHMARKS.md measured on this machine.</div>
 </div>
 <div class="card">
  <h2>All recorded runs <small>benchmarks/results.jsonl + server_runs.jsonl</small></h2>
  <div style="max-height:430px;overflow:auto"><table id="btable"><tr><th>src</th><th>digits</th><th>backend</th>
   <th>total s</th><th>split s</th><th>convert s</th><th>RSS MB</th><th>digits/s</th></tr></table></div>
 </div>
</section>

</main>
<div id="toast"></div>
<script>
"use strict";
function $(id){return document.getElementById(id);}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){
 return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function fmt(v,d){if(v==null||isNaN(v))return '-';return (+v).toFixed(d==null?3:d);}
function nf(v){if(v==null||isNaN(v))return '-';return (+v).toLocaleString('en-US');}
function toast(msg,bad){var t=document.createElement('div');t.className='tst'+(bad?' bad':'');
 t.textContent=msg;$('toast').appendChild(t);setTimeout(function(){t.remove();},4200);}
async function jget(u){var r=await fetch(u);if(!r.ok)throw new Error(u+' -> '+r.status);return r.json();}

document.querySelectorAll('nav button').forEach(function(b){b.onclick=function(){
 document.querySelectorAll('nav button').forEach(function(x){x.classList.toggle('act',x===b);});
 document.querySelectorAll('main>section').forEach(function(s){s.classList.toggle('act',s.id==='tab-'+b.dataset.tab);});
 if(b.dataset.tab==='map')drawMap();
 if(b.dataset.tab==='bench')loadBench();
 if(b.dataset.tab==='digits')fillJobsSel();
};});

async function health(){try{var h=await jget('/health');
 $('hdot').className='dot on';
 $('htxt').textContent='engine online'+(h.cuda?' + cuda':'')+' | '+h.active+' conn';}
 catch(e){$('hdot').className='dot off';$('htxt').textContent='engine offline';}}
setInterval(health,4000);health();

var curId=null,es=null,digitsCache={};
var PHASES=[['split','binary splitting',0.66],['sqrt','sqrt(10005)',0.05],['final','final combine',0.08],
 ['convert','decimal conversion',0.13],['bbp','BBP verify',0.06],['reference','reference check',0.02],
 ['done','complete',1]];
function overall(s){
 if(s.state==='queued')return 0;
 for(var i=0;i<PHASES.length;i++)if(PHASES[i][0]===s.phase){
  var base=0;for(var k=0;k<i;k++)base+=PHASES[k][2];
  return Math.min(1,base+PHASES[i][2]*s.frac);}
 return s.frac*0.6;}
function rateOf(s){return s.total_sec>0?s.digits/s.total_sec:null;}
function renderLive(s){
 var pct=(overall(s)*100).toFixed(1);
 $('pbig').style.width=pct+'%';
 var ph=null;PHASES.forEach(function(p){if(p[0]===s.phase)ph=p;});
 $('ptxt').innerHTML=s.state==='queued'?'<span class="k">queued</span>':
  s.state==='failed'?'<span class="bad">failed: '+esc(s.error)+'</span>':
  '<span class="phase">'+esc(ph?ph[1]:s.phase)+'</span> | '+(s.frac*100).toFixed(1)+
  '% of phase | '+pct+'% overall';}
function prettyDigits(s){
 if(s.digits<=70&&s.head)return esc(s.head);
 return esc(s.head||'')+'\u2026'+esc(s.tail||'');}
function renderResult(s){
 $('rid').textContent='#'+s.id;
 var vr=s.verify?(s.verify_fails===0?'<span class="ok">BBP PASS</span>':
  (s.verify_fails>0?'<span class="bad">BBP FAIL ('+s.verify_fails+')</span>':'<span class="k">pending</span>')):'<span class="k">off</span>';
 if(s.state==='failed'){$('result').innerHTML='<span class="bad">'+esc(s.error)+'</span>';return;}
 if(s.state!=='done'){$('result').innerHTML='<div class="k">running\u2026 (result appears on completion)</div>';return;}
 $('result').innerHTML='<div class="big">'+nf(s.digits)+' digits</div>'+
  '<div class="pre" style="margin:10px 0">'+prettyDigits(s)+'</div><div class="kv">'+
  '<div><div class="k">total</div><div class="v">'+fmt(s.total_sec)+' s</div></div>'+
  '<div><div class="k">digits / s</div><div class="v">'+nf(Math.round(rateOf(s)||0))+'</div></div>'+
  '<div><div class="k">binary split</div><div class="v">'+fmt(s.bs)+' s</div></div>'+
  '<div><div class="k">convert</div><div class="v">'+fmt(s.convert)+' s</div></div>'+
  '<div><div class="k">sqrt</div><div class="v">'+fmt(s.sqrt)+' s</div></div>'+
  '<div><div class="k">final</div><div class="v">'+fmt(s.final)+' s</div></div>'+
  '<div><div class="k">peak RSS</div><div class="v">'+fmt(s.peak_rss_mb,1)+' MB</div></div>'+
  '<div><div class="k">verify</div><div class="v">'+vr+'</div></div></div>'+
  (s.digits_kept?'<div style="margin-top:10px"><button class="btn ghost" onclick="gotoDigits('+s.id+')">Open in Digits tab \u2192</button></div>':
   '<div class="note">decimal string too large to keep in server RAM; use CLI --out for the full expansion</div>');}
function gotoDigits(id){fillJobsSel().then(function(){$('djob').value=id;loadStats();});}

async function runJob(){
 var body={digits:+$('digits').value,backend:$('backend').value,leaf:+$('leaf').value,
  guard:+$('guard').value,verify:$('verify').checked};
 $('gorun').disabled=true;
 try{
  var r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(body)});
  if(!r.ok){var e=await r.json().catch(function(){return {error:'HTTP '+r.status};});
   throw new Error(e.error||'submit failed');}
  var d=await r.json(),id=d.id;curId=id;
  $('live').innerHTML='<div class="k">Job #'+id+' submitted ('+nf(body.digits)+' digits, '+esc(body.backend)+').</div>';
  if(es)es.close();
  es=new EventSource('/api/stream?id='+id);
  es.onmessage=function(ev){var s=JSON.parse(ev.data);renderLive(s);
   if(s.state==='done'||s.state==='failed'){es.close();es=null;$('gorun').disabled=false;
    renderResult(s);refreshJobs();
    toast('job #'+s.id+(s.state==='done'?' done in '+fmt(s.total_sec)+'s':' failed'),s.state!=='done');}};
  es.onerror=function(){if(es){es.close();es=null;}$('gorun').disabled=false;refreshJobs();};
 }catch(err){toast(String(err.message||err),true);$('gorun').disabled=false;}}

async function refreshJobs(){
 try{
  var js=await jget('/api/jobs'),tb=$('jobs');
  while(tb.rows.length>1)tb.deleteRow(1);
  js.slice().reverse().forEach(function(s){
   var tr=tb.insertRow();
   tr.onclick=function(){curId=s.id;renderLive(s);renderResult(s);
    tb.querySelectorAll('tr').forEach(function(x){x.classList.remove('sel');});tr.classList.add('sel');};
   var cls=s.state==='done'?'ok':(s.state==='failed'?'bad':'');
   tr.innerHTML='<td>'+s.id+'</td><td>'+nf(s.digits)+'</td><td>'+esc(s.backend)+'</td>'+
    '<td class="'+cls+'">'+esc(s.state)+'</td><td>'+esc(s.phase)+'</td>'+
    '<td>'+fmt(s.total_sec)+'</td><td>'+nf(Math.round(rateOf(s)||0))+'</td>'+
    '<td>'+fmt(s.peak_rss_mb,1)+'</td><td>'+(s.verify&&s.state==='done'?(s.verify_fails?'<span class="bad">FAIL</span>':'<span class="ok">PASS</span>'):'-')+'</td>';});
 }catch(e){}}
setInterval(refreshJobs,1500);

/* ---------- device mapping DAG ---------- */
async function drawMap(){
 var D=+$('mdigits').value||1000000,b=$('mbackend').value,m;
 try{m=await jget('/api/mapping?digits='+D+'&backend='+b);}catch(e){toast('mapping failed',true);return;}
 $('mapinfo').innerHTML='terms N = <b class="mono">'+nf(m.terms)+'</b> | '+m.levels+
  ' splitting levels | planned '+nf(m.summary.planned_est_ms)+' ms vs CPU-only '+nf(m.summary.cpu_only_est_ms)+
  ' ms (model speedup <b>'+m.summary.est_speedup.toFixed(2)+'\u00d7</b>) | <b>'+esc(m.model)+'</b>'+
  (m.gpu_available?'':' | <span class="warn">backend=cpu: every node maps to CPU</span>');
 $('mapnote').textContent='DAG: one column per splitting level; the coloured blocks are the (count) products that '+
  'level performs, batched. Level 0 = leaf products over '+m.layers[0].span_terms+' terms (~60-bit factors); '+
  'the rightmost level is the root. Device colour comes from the scheduler cost model: a level runs on the GPU only '+
  'when modelled NTT + PCIe time < 80% of the CPU estimate (and, for hybrid, only in the upper tree). '+
  'GPU constants are placeholders until experiment E4 calibrates the CUDA NTT backend.';
 var svg=$('dag'),L=m.layers,W=1180,H=540,mL=60,mR=60,step=(W-mL-mR)/Math.max(1,L.length-1);
 function dev(d){return d==='gpu'?'#ff9440':'#4f8cff';}
 var out='';
 L.forEach(function(l,li){
  var x=mL+li*step,shown=Math.min(l.count,14);
  var gap=Math.min(26,(H-170)/Math.max(shown,1));
  var hgt=Math.max(6,Math.min(24,gap-2));
  var ytop=H/2-(shown-1)*gap/2;
  for(var i=0;i<shown;i++){var y=ytop+i*gap;
   out+='<rect x="'+(x-10)+'" y="'+(y-hgt/2)+'" width="20" height="'+hgt+'" rx="4" fill="'+dev(l.device)+'" opacity="0.92"/>';}
  if(l.count>shown)out+='<text x="'+x+'" y="'+(ytop+shown*gap+6)+'" font-size="10" fill="#8b98b3" text-anchor="middle">\u00d7'+nf(l.count)+'</text>';
  out+='<text x="'+x+'" y="24" font-size="11" fill="#dbe4f5" text-anchor="middle">L'+l.level+'</text>';
  out+='<text x="'+x+'" y="40" font-size="10" fill="#8b98b3" text-anchor="middle">'+
   (l.node_bits>1e6?(l.node_bits/1e6).toFixed(2)+'Mb':(l.node_bits/1e3).toFixed(0)+'kb')+'</text>';
  out+='<text x="'+x+'" y="'+(H-34)+'" font-size="10" fill="'+dev(l.device)+'" text-anchor="middle" font-weight="bold">'+l.device+'</text>';
  out+='<text x="'+x+'" y="'+(H-18)+'" font-size="10" fill="#8b98b3" text-anchor="middle">'+
   (l.device==='gpu'?(l.est_gpu_ms+l.est_xfer_ms).toFixed(1):l.est_cpu_ms.toFixed(1))+'ms</text>';
  if(li<L.length-1){var nx=mL+(li+1)*step;
   out+='<path d="M'+(x+12)+' '+(H/2-46)+' Q'+((x+nx)/2)+' '+(H/2-70)+' '+(nx-12)+' '+(H/2-46)+'" stroke="#2a3b5e" fill="none" stroke-width="1.2"/>';
   out+='<path d="M'+(x+12)+' '+(H/2+46)+' Q'+((x+nx)/2)+' '+(H/2+70)+' '+(nx-12)+' '+(H/2+46)+'" stroke="#2a3b5e" fill="none" stroke-width="1.2"/>';}});
 out+='<text x="'+(W/2)+'" y="'+(H-2)+'" font-size="11" fill="#8b98b3" text-anchor="middle">binary-splitting tree: leaves (left, ~60-bit terms) \u2192 root (right, \u2248 D\u00b73.32 bits) \u00b7 colour = scheduled device \u00b7 label = batch cost estimate</text>';
 svg.innerHTML=out;
 var tb=$('maptbl');while(tb.rows.length>1)tb.deleteRow(1);
 L.forEach(function(l){var tr=tb.insertRow();
  tr.innerHTML='<td>'+l.level+'</td><td>'+nf(l.span_terms)+'</td><td>'+nf(l.count)+'</td>'+
   '<td>'+nf(Math.round(l.node_bits))+'</td><td style="color:'+dev(l.device)+'">'+l.device+'</td>'+
   '<td>'+l.est_cpu_ms.toFixed(3)+'</td><td>'+l.est_gpu_ms.toFixed(3)+'</td><td>'+l.est_xfer_ms.toFixed(3)+'</td>';});
}

/* ---------- digits ---------- */
async function fillJobsSel(){
 try{var js=await jget('/api/jobs'),sel=$('djob'),old=sel.value;
  sel.innerHTML='';
  js.filter(function(s){return s.state==='done'&&s.digits_kept;}).forEach(function(s){
   var o=document.createElement('option');o.value=s.id;
   o.textContent='#'+s.id+' \u00b7 '+nf(s.digits)+' digits \u00b7 '+s.backend;sel.appendChild(o);});
  if(!sel.options.length){var o=document.createElement('option');
   o.textContent='(no jobs with kept digits \u2014 run one \u226410M digits)';sel.appendChild(o);}
  if(old&&sel.querySelector('option[value="'+old+'"]'))sel.value=old;
 }catch(e){}}
function parseDigits(s){s=s.replace(/^3\./,'').replace(/\s+/g,'');return s;}
async function getDigits(id){
 if(digitsCache[id])return digitsCache[id];
 var r=await fetch('/api/job?id='+id+'&what=digits');
 if(!r.ok)throw new Error('no digits for that job');
 var s=parseDigits(await r.text());digitsCache[id]=s;return s;}
function drawFreq(st,tot){
 var svg=$('freq'),W=1140,H=250,gap=W/10,bw=gap*0.62,mx=Math.max.apply(null,st.freq);
 var out='';
 for(var d=0;d<10;d++){
  var h=(st.freq[d]/mx)*(H-76),x=d*gap+(gap-bw)/2,y=H-50-h;
  var devp=(st.freq[d]/tot-0.1)/0.1*100;
  out+='<rect x="'+x+'" y="26" width="'+bw+'" height="'+(H-76)+'" fill="#0b1120" stroke="#1e2a42"/>';
  out+='<rect x="'+x+'" y="'+y+'" width="'+bw+'" height="'+h+'" rx="3" fill="'+(Math.abs(devp)>0.15?'#ffc857':'#4f8cff')+'"/>';
  out+='<text x="'+(x+bw/2)+'" y="'+(y-6)+'" font-size="10" fill="#dbe4f5" text-anchor="middle">'+nf(st.freq[d])+'</text>';
  out+='<text x="'+(x+bw/2)+'" y="'+(H-34)+'" font-size="13" fill="#dbe4f5" text-anchor="middle">'+d+'</text>';
  out+='<text x="'+(x+bw/2)+'" y="'+(H-16)+'" font-size="10" fill="#8b98b3" text-anchor="middle">'+(st.freq[d]/tot*100).toFixed(3)+'%</text>';
  out+='<text x="'+(x+bw/2)+'" y="'+(H-4)+'" font-size="9" fill="'+(Math.abs(devp)>0.15?'#ffc857':'#2fd57b')+'" text-anchor="middle">'+(devp>=0?'+':'')+devp.toFixed(2)+'%</text>';}
 out+='<text x="4" y="16" font-size="11" fill="#8b98b3">digit frequencies (last column: relative deviation from uniform 10%, green if |\u0394|<0.15%)</text>';
 svg.innerHTML=out;
 $('statinfo').textContent=nf(tot)+' fractional digits analysed.'+
  (st.total?' (job reports '+nf(st.total)+')':'');}
async function loadStats(){
 var id=$('djob').value;if(!id)return;
 try{var st=await jget('/api/job?id='+id+'&what=stats');
  drawFreq(st,st.freq.reduce(function(a,b){return a+b;},0));}
 catch(e){$('freq').innerHTML='';$('statinfo').textContent='no stats available';}}
async function findPat(){
 var id=$('djob').value,p=$('dpattern').value;if(!id||!p){toast('pick a job and a pattern',true);return;}
 $('findres').textContent='searching '+nf(p.length)+'-char pattern\u2026';
 try{var s=await getDigits(id),hits=[],i=-1,t0=performance.now();
  while((i=s.indexOf(p,i+1))!==-1&&hits.length<50)hits.push(i);
  var ms=performance.now()-t0;
  $('findres').textContent=hits.length?('first '+(hits.length===50?'50+ ':'')+hits.length+
   ' hit(s): #'+hits.slice(0,8).join(', #')+(hits.length>=50?' \u2026':''))+' in '+ms.toFixed(0)+'ms':
   'no occurrence in '+nf(s.length)+' digits ('+ms.toFixed(0)+'ms)';
 }catch(err){$('findres').textContent=String(err.message||err);}}
async function loadWindow(){
 var id=$('djob').value,off=+$('doff').value||0;if(!id){toast('pick a job first',true);return;}
 try{
  var r=await fetch('/api/job?id='+id+'&what=digits&offset='+off+'&max=10000');
  if(!r.ok)throw new Error('window unavailable');
  var s=await r.text(),lines=[];
  for(var i=0;i<s.length;i+=1000)lines.push((off+i+1)+' | '+s.slice(i,i+1000));
  $('dwin').textContent=lines.join('\n')||'(empty)';
 }catch(err){$('dwin').textContent=String(err.message||err);}}

/* ---------- benchmarks ---------- */
var YCR=[{d:10000000,t:0.148},{d:100000000,t:1.107},{d:1000000000,t:14.670}];
function collectRuns(){
 var rows=[];
 window._cli.forEach(function(r){rows.push({src:'cli',digits:r.digits,backend:r.backend||'cpu',
  total:r.median_total,bs:r.bs,cv:r.convert,rss:r.peak_rss_mb});});
 window._srv.forEach(function(r){rows.push({src:'ui',digits:r.digits,backend:r.backend||'cpu',
  total:r.total_sec,bs:r.bs,cv:r.convert,rss:r.peak_rss_mb});});
 return rows;}
function drawBench(rows){
 var svg=$('bchart'),W=1140,H=430,L=80,R=30,T=34,B=54;
 var all=rows.map(function(r){return [r.digits,r.total];});
 YCR.forEach(function(y){all.push([y.d,y.t]);});
 var xs=all.map(function(p){return p[0];}),ys=all.map(function(p){return p[1];});
 var lx0=Math.log10(Math.min.apply(null,xs)/2),lx1=Math.log10(Math.max.apply(null,xs)*2);
 var ly0=Math.log10(Math.min.apply(null,ys)/2),ly1=Math.log10(Math.max.apply(null,ys)*3);
 function X(d){return L+(W-L-R)*(Math.log10(d)-lx0)/(lx1-lx0);}
 function Y(t){return H-B-(H-T-B)*(Math.log10(t)-ly0)/(ly1-ly0);}
 var out='';
 for(var gx=Math.ceil(lx0);gx<=Math.floor(lx1);gx++){var x=X(Math.pow(10,gx));
  out+='<line x1="'+x+'" y1="'+T+'" x2="'+x+'" y2="'+(H-B)+'" stroke="#1e2a42"/>';
  var lab=gx>=9?(Math.pow(10,gx-6)>=1e6?(Math.pow(10,gx-6))+'M':Math.pow(10,gx-9)+'G'):
   (gx>=6?Math.pow(10,gx-3)+'k':Math.pow(10,gx));
  out+='<text x="'+x+'" y="'+(H-B+18)+'" font-size="11" fill="#8b98b3" text-anchor="middle">'+lab+'</text>';}
 for(var gy=Math.ceil(ly0);gy<=Math.floor(ly1);gy++){var y=Y(Math.pow(10,gy));
  out+='<line x1="'+L+'" y1="'+y+'" x2="'+(W-R)+'" y2="'+y+'" stroke="#1e2a42"/>';
  out+='<text x="'+(L-8)+'" y="'+(y+4)+'" font-size="11" fill="#8b98b3" text-anchor="end">'+
   (Math.pow(10,gy)<1?Math.pow(10,gy):nf(Math.pow(10,gy)))+'s</text>';}
 YCR.forEach(function(y){var x=X(y.d),y0=Y(y.t);
  out+='<line x1="'+x+'" y1="'+(H-B)+'" x2="'+x+'" y2="'+y0+'" stroke="#2fd57b" stroke-width="10" opacity="0.22"/>';
  out+='<circle cx="'+x+'" cy="'+y0+'" r="5" fill="#2fd57b"/>';
  out+='<text x="'+x+'" y="'+(y0-10)+'" font-size="10" fill="#2fd57b" text-anchor="middle">y-cruncher '+y.t+'s</text>';});
 rows.forEach(function(r){var c=r.backend==='cpu'?'#4f8cff':'#ff9440';
  out+='<circle cx="'+X(r.digits)+'" cy="'+Y(r.total)+'" r="4" fill="'+c+'" opacity="0.85">'+
   '<title>'+r.src+' '+r.backend+' '+nf(r.digits)+' digits: '+fmt(r.total)+'s</title></circle>';});
 out+='<text x="'+((L+W-R)/2)+'" y="'+(H-8)+'" font-size="11" fill="#8b98b3" text-anchor="middle">digits (log)</text>';
 out+='<text x="16" y="'+(T-12)+'" font-size="11" fill="#8b98b3">end-to-end seconds (log)</text>';
 svg.innerHTML=out;}
async function loadBench(){
 try{
  window._cli=await jget('/api/benchmarks');
  window._srv=await jget('/api/history');
 }catch(e){window._cli=[];window._srv=[];}
 var rows=collectRuns(),tb=$('btable');
 while(tb.rows.length>1)tb.deleteRow(1);
 rows.sort(function(a,b){return b.digits-a.digits||a.total-b.total;}).forEach(function(r){
  var tr=tb.insertRow();
  tr.innerHTML='<td>'+r.src+'</td><td>'+nf(r.digits)+'</td><td>'+esc(r.backend)+'</td>'+
   '<td>'+fmt(r.total)+'</td><td>'+fmt(r.bs)+'</td><td>'+fmt(r.cv)+'</td>'+
   '<td>'+fmt(r.rss,1)+'</td><td>'+nf(Math.round(r.digits/(r.total||1)))+'</td>';});
 drawBench(rows);}

refreshJobs();
</script>
</body>
</html>
)====";
