(function(){
const D=window.DATA;
const LV=["NORMAL","WATCH","SCHEDULE","URGENT"],LC=["--mut","--warn","--sched","--crit"];
const el=id=>document.getElementById(id);
const hd=a=>"<tr>"+a.map(x=>`<th>${x}</th>`).join("")+"</tr>";
const set=(id,v)=>{const e=el(id);if(e)e.textContent=v};

set("m-range",D.range[0]+" to "+D.range[1]);set("m-n",D.n.toLocaleString());
set("ae-ep",D.ae.epochs);set("ae-best",D.ae.best);set("thr",D.threshold.toFixed(4));

/* verdict cards */
const C=D.clf.pooled,R=D.rul.pooled,skill=(1-R.mae/R.baseline_mae)*100;
const flagged=(100-D.mix.failure[0]).toFixed(0),leave=(100-D.mix.healthy[0]).toFixed(0);
set("fp",leave+"% of healthy windows");
el("verdicts").innerHTML=[
 ["Anomaly detector",`<span class="pill ok">works</span>`,flagged+"%","of in-failure windows raise an alarm",`${leave}% of healthy windows also leave NORMAL.`],
 ["Failure classifier",`<span class="pill bad">no skill</span>`,C.roc_auc.toFixed(2),"pooled ROC-AUC (0.50 = chance)",`F1 ${C.f1.toFixed(2)}, precision ${C.precision.toFixed(2)}, recall ${C.recall.toFixed(2)}.`],
 ["RUL estimator",`<span class="pill bad">no skill</span>`,(skill>0?"+":"")+skill.toFixed(0)+"%","error vs a constant guess",`MAE ${R.mae.toFixed(1)} h vs ${R.baseline_mae.toFixed(1)} h for the baseline.`]
].map(c=>`<div class="v"><div class="top"><b>${c[0]}</b>${c[1]}</div><div class="big">${c[2]}</div><div>${c[3]}</div><small>${c[4]}</small></div>`).join("");
set("clfAuc",C.roc_auc.toFixed(2));set("rulMae",R.mae.toFixed(1)+" h");set("rulBase",R.baseline_mae.toFixed(1)+" h");

/* static tables */
const sig=[["TP2","analogue","Compressor pressure"],["TP3","analogue","Pressure at the pneumatic panel"],["H1","analogue","Pressure drop when the cyclone separator filter discharges"],
["DV_pressure","analogue","Pressure drop when the towers discharge air dryers"],["Reservoirs","analogue","Pressure downstream of the reservoirs"],["Oil_temperature","analogue","Compressor oil temperature"],
["Motor_current","analogue","Motor current"],["COMP","analogue","Air intake valve signal"],["DV_eletric","digital","Compressor outlet valve"],["Towers","digital","Which dryer tower is active"],
["MPG","digital","Compressor load start under low pressure"],["LPS","digital","Low-pressure signal"],["Pressure_switch","digital","Discharge pressure detector"],["Oil_level","digital","Oil level"],["Caudal_impulses","digital","Air flow pulses"]];
el("sigT").innerHTML=hd(["Signal","Type","Meaning"])+sig.map(s=>`<tr><td class="m">${s[0]}</td><td>${s[1]}</td><td>${s[2]}</td></tr>`).join("");
const fdur=(a,b)=>{const h=(new Date(b.replace(" ","T")+"Z")-new Date(a.replace(" ","T")+"Z"))/36e5;return h.toFixed(1)+" h"};
el("failT").innerHTML=hd(["","Start","End","Duration"])+D.failures.map((f,i)=>`<tr><td>F${i+1}</td><td class="m">${f[0]}</td><td class="m">${f[1]}</td><td class="m">${fdur(f[0],f[1])}</td></tr>`).join("");

/* SVG helpers */
const W=1000,ML=46,MR=10;
const T=s=>new Date(s+":00Z").getTime();
function xs(ts){const a=T(ts[0]),b=T(ts[ts.length-1]);return t=>ML+(T(t)-a)/(b-a)*(W-ML-MR)}
function bands(x,y0,y1,t0,t1){return D.failures.map(([s,e])=>{const a=s.slice(0,16).replace(" ","T"),b=e.slice(0,16).replace(" ","T");
 if(T(b)<T(t0)||T(a)>T(t1))return"";const xa=x(T(a)<T(t0)?t0:a),xb=x(T(b)>T(t1)?t1:b);
 return`<rect x="${xa}" y="${y0}" width="${Math.max(2,xb-xa)}" height="${y1-y0}" fill="var(--band)"/>`}).join("")}
function dateTicks(ts,x,y,step,fmt){const a=T(ts[0]),b=T(ts[ts.length-1]);let out="";
 for(let t=Math.ceil(a/step)*step;t<=b;t+=step){const d=new Date(t),iso=d.toISOString().slice(0,16),px=x(iso);
  out+=`<line x1="${px}" x2="${px}" y1="${y-4}" y2="${y}" stroke="var(--line)"/><text x="${px}" y="${y+14}" text-anchor="middle">${fmt(d)}</text>`}return out}
const mon=d=>d.toLocaleString("en",{month:"short",day:"numeric",timeZone:"UTC"});
const hr=d=>d.toLocaleString("en",{month:"short",day:"numeric",hour:"2-digit",hour12:false,timeZone:"UTC"}).replace(",","");
function legend(id,items){el(id).innerHTML=items.map(([c,t])=>`<span><i style="background:${c}"></i>${t}</span>`).join("")}
const lvLegend=()=>LV.map((n,i)=>[`var(${LC[i]})`,n]);
legend("lg1",[...lvLegend(),["var(--band)","failure window"]]);
legend("lg2",[...lvLegend(),["var(--band)","failure window"],["var(--acc)","RUL predicted"],["var(--fg)","RUL true"]]);

function scoreChart(S,o){
 const x=xs(S.t),H=o.h,top=8,bot=H-34,strip=14,plotB=bot-strip-6;
 const lo=Math.log10(0.01),hi=Math.log10(Math.max(10,...S.s)),y=v=>top+(1-(Math.log10(Math.max(v,.01))-lo)/(hi-lo))*(plotB-top);
 let g="";[0.01,0.1,1,10,100].filter(v=>Math.log10(v)<=hi).forEach(v=>{g+=`<line x1="${ML}" x2="${W-MR}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${ML-6}" y="${y(v)+4}" text-anchor="end">${v}</text>`});
 const pts=S.t.map((t,i)=>`${i?"L":"M"}${x(t).toFixed(1)},${y(S.s[i]).toFixed(1)}`).join("");
 const w=Math.max(1.5,(W-ML-MR)/S.t.length);
 const st=S.t.map((t,i)=>S.l[i]?`<rect x="${(x(t)-w/2).toFixed(1)}" y="${bot-strip}" width="${w.toFixed(1)}" height="${strip}" fill="var(${LC[S.l[i]]})"/>`:"").join("");
 return`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label}">${bands(x,top,bot,S.t[0],S.t[S.t.length-1])}${g}
 <line x1="${ML}" x2="${W-MR}" y1="${y(1)}" y2="${y(1)}" stroke="var(--crit)" stroke-dasharray="4 3"/>
 <path d="${pts}" fill="none" stroke="var(--acc)" stroke-width="1.2"/>
 <rect x="${ML}" y="${bot-strip}" width="${W-ML-MR}" height="${strip}" fill="var(--chip)"/>${st}${dateTicks(S.t,x,bot,o.step,o.fmt)}</svg>`}
function rulChart(S){
 const x=xs(S.t),H=170,top=8,bot=H-28,y=v=>top+(1-v/72)*(bot-top);
 let g="";[0,24,48,72].forEach(v=>{g+=`<line x1="${ML}" x2="${W-MR}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${ML-6}" y="${y(v)+4}" text-anchor="end">${v}h</text>`});
 const line=k=>{let d="",pen=false;S.t.forEach((t,i)=>{const v=S[k][i];if(v==null){pen=false;return}d+=`${pen?"L":"M"}${x(t).toFixed(1)},${y(v).toFixed(1)}`;pen=true});return d};
 return`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted versus true remaining useful life">${bands(x,top,bot,S.t[0],S.t[S.t.length-1])}${g}
 <path d="${line("rt")}" fill="none" stroke="var(--fg)" stroke-width="1.6"/><path d="${line("rp")}" fill="none" stroke="var(--acc)" stroke-width="1.4"/>${dateTicks(S.t,x,bot,6*3600e3,hr)}</svg>`}

/* generic small line chart: series [{name,color,v:[...]}], x labels */
function lineChart(xl,series,o){
 const w=500,h=o.h||200,l=44,r=10,t=10,b=28,max=o.max,min=o.min||0,n=xl.length;
 const px=i=>l+(n===1?0:i/(n-1))*(w-l-r),py=v=>t+(1-(v-min)/(max-min))*(h-t-b);
 let g="";(o.ticks).forEach(v=>{g+=`<line x1="${l}" x2="${w-r}" y1="${py(v)}" y2="${py(v)}" stroke="var(--grid)"/><text x="${l-6}" y="${py(v)+4}" text-anchor="end">${v}</text>`});
 if(o.ref!=null)g+=`<line x1="${l}" x2="${w-r}" y1="${py(o.ref)}" y2="${py(o.ref)}" stroke="var(--crit)" stroke-dasharray="4 3"/>`;
 const lab=xl.map((s,i)=>(o.every?i%o.every===0:true)?`<text x="${px(i)}" y="${h-8}" text-anchor="middle">${s}</text>`:"").join("");
 const paths=series.map(s=>`<path d="${s.v.map((v,i)=>`${i?"L":"M"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join("")}" fill="none" stroke="${s.color}" stroke-width="1.8"/>`+
   (n<=8?s.v.map((v,i)=>`<circle cx="${px(i)}" cy="${py(v)}" r="3" fill="${s.color}"/>`).join(""):"")).join("");
 return`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${o.label}">${g}${paths}${lab}</svg>`}

el("full").innerHTML=scoreChart(D.full,{h:260,step:7*86400e3,fmt:mon,label:"Anomaly score over time"});
el("aeC").innerHTML=lineChart(D.ae.loss.map((_,i)=>i+1),[{name:"train",color:"var(--acc)",v:D.ae.loss},{name:"val",color:"var(--sched)",v:D.ae.val}],
 {max:Math.ceil(Math.max(...D.ae.loss)*20)/20,ticks:[0,0.05,0.1,0.15],every:10,label:"Autoencoder training and validation loss"});
legend("lgAE",[["var(--acc)","training"],["var(--sched)","validation"]]);
el("sweepC").innerHTML=lineChart(D.sweep.map(s=>s.lookahead_h+"h"),[{color:"var(--acc)",v:D.sweep.map(s=>s.roc_auc)}],{max:1,ticks:[0,.25,.5,.75,1],ref:.5,label:"ROC-AUC by lookahead horizon, dashed line is chance"});

let cur=0;
function drawZoom(){
 const z=D.zoom[cur];
 el("zoomScore").innerHTML=scoreChart(z,{h:210,step:6*3600e3,fmt:hr,label:"Anomaly score around failure"});
 el("zoomRul").innerHTML=rulChart(z);
 const s=D.summary[cur],lead=s.first_SCHEDULE_h_before_start;
 el("zoomNote").textContent=`Failure ${s.failure} starts ${z.start.replace("T"," ")}. `+(lead==null?"The scheduler never reached SCHEDULE_MAINTENANCE in this period.":lead>0?`SCHEDULE_MAINTENANCE was reached ${lead.toFixed(1)} h before it started.`:`SCHEDULE_MAINTENANCE was reached ${Math.abs(lead).toFixed(1)} h after it started.`)+" Lower chart: predicted RUL (blue) against true RUL (black), shown only inside the 72 h pre-failure zone.";
 [...el("tabs").children].forEach((b,i)=>b.setAttribute("aria-pressed",i===cur));
}
el("tabs").innerHTML=D.zoom.map((z,i)=>`<button type="button" data-i="${i}">F${i+1} · ${z.start.slice(0,10)}</button>`).join("");
el("tabs").addEventListener("click",e=>{const b=e.target.closest("button");if(b){cur=+b.dataset.i;drawZoom()}});
drawZoom();

/* tables */
const aucC=v=>v<.5?"neg":v>=.7?"pos":"";
el("clfT").innerHTML=hd(["Held-out","ROC-AUC","F1","Precision","Recall"])+["F1","F2","F3","F4","pooled"].map(k=>{const r=D.clf[k];
 return`<tr><td>${k==="pooled"?"<b>pooled</b>":k}</td><td class="m ${aucC(r.roc_auc)}">${r.roc_auc.toFixed(2)}</td><td class="m">${r.f1.toFixed(2)}</td><td class="m">${r.precision.toFixed(2)}</td><td class="m">${r.recall.toFixed(2)}</td></tr>`}).join("");
el("rulT").innerHTML=hd(["Held-out","Model MAE (h)","Constant MAE (h)","Within ±6 h","Rank corr."])+["F1","F2","F3","F4","pooled"].map(k=>{const r=D.rul[k];
 return`<tr><td>${k==="pooled"?"<b>pooled</b>":k}</td><td class="m ${r.mae<r.baseline_mae?"pos":"neg"}">${r.mae.toFixed(1)}</td><td class="m">${r.baseline_mae.toFixed(1)}</td><td class="m">${(r.within_6h*100).toFixed(0)}%</td><td class="m ${r.spearman<0?"neg":"pos"}">${r.spearman.toFixed(2)}</td></tr>`}).join("");
const hs=[1,3,6,12,24],dv=[["absolute (baseline)","Absolute features"],["original + drift features","Absolute + drift features"],["drift-free only","Drift-free features only"]];
el("driftT").innerHTML=hd(["Feature set",...hs.map(h=>h+"h")])+dv.map(([k,n])=>`<tr><td>${n}</td>${hs.map(h=>{const v=D.drift[k][h];return`<td class="m ${v<.5?"neg":""}">${v.toFixed(2)}</td>`}).join("")}</tr>`).join("");
const mx=Math.max(...D.shap.map(s=>s.mean_abs_shap));
el("shap").innerHTML=D.shap.map(s=>`<div class="bar"><span title="${s.feature}">${s.feature}</span><span class="t"><i style="width:${s.mean_abs_shap/mx*100}%"></i></span><span>${s.mean_abs_shap.toFixed(2)}</span></div>`).join("");

el("leadT").innerHTML=hd(["Failure","Schedule (h before)","Urgent (h before)","24 h before ≥ SCHEDULE","During ≥ SCHEDULE"])+
 D.summary.map(s=>{const c=v=>v==null?'<td class="m dim">never</td>':`<td class="m ${v>0?"pos":"neg"}">${v>0?"+":""}${v.toFixed(1)}</td>`;
 return`<tr><td>${s.failure}</td>${c(s.first_SCHEDULE_h_before_start)}${c(s.first_URGENT_h_before_start)}<td class="m">${(s.pre24h_ge_SCHEDULE*100).toFixed(0)}%</td><td class="m">${(s.in_failure_ge_SCHEDULE*100).toFixed(0)}%</td></tr>`}).join("");
el("mix").innerHTML=[["Healthy windows",D.mix.healthy],["In-failure windows",D.mix.failure]].map(([n,m])=>
 `<div style="margin:6px 0 12px"><div class="mono" style="margin-bottom:4px;background:none;padding:0">${n}</div>
 <div style="display:flex;height:22px;border-radius:3px;overflow:hidden;background:var(--chip)">${m.map((v,i)=>`<div title="${LV[i]} ${v}%" style="width:${v}%;background:var(${LC[i]});opacity:${i?1:.35}"></div>`).join("")}</div>
 <div class="legend">${m.map((v,i)=>`<span><i style="background:var(${LC[i]})"></i>${LV[i]} ${v.toFixed(1)}%</span>`).join("")}</div></div>`).join("")+
 '<p class="note">Healthy windows should be NORMAL. The share that is not is the false-alarm cost of the detector.</p>';
const M=D.matrix,LI={NORMAL:0,WATCH:1,SCHEDULE_MAINTENANCE:2,URGENT:3};
el("matrix").innerHTML=hd(["Likelihood \\ consequence",...M.cols])+M.rows.map((r,i)=>`<tr><td>${r}</td>${M.cells[i].map(c=>`<td class="cell L${LI[c]}">${c.replace("_MAINTENANCE","")}</td>`).join("")}</tr>`).join("");

/* active section in the contents list */
const links=[...document.querySelectorAll("nav.toc a")];
if("IntersectionObserver" in window){
 const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)links.forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+e.target.id))}),{rootMargin:"-20% 0px -70% 0px"});
 document.querySelectorAll("main [id]").forEach(s=>io.observe(s));
}
})();
