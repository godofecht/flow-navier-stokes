"use strict";
const $=id=>document.getElementById(id);
let datasets={},scenario="nonlinear",observable="vorticity",frameIndex=0,playing=false,timer=null,stage=0;
const descriptions=[
"Compute the velocity derivatives from the current state: −(u·∇)u + ν∇²u. This is the actual momentum right-hand side evaluated by the Python solver.",
"Form a provisional RK2 velocity using the current derivative. The solver projects this predictor to suppress discrete divergence.",
"Solve a periodic Poisson equation for pressure with conjugate gradients, then subtract its gradient. The matched discrete operators enforce incompressibility.",
"Evaluate the right-hand side at the predictor, average both RK2 derivatives and project again. The displayed frame is the completed, projected state."
];
function current(){return datasets[scenario].frames[frameIndex]}
function number(x){return Number(x).toExponential(3)}
function color(value,range){const z=Math.max(-1,Math.min(1,value/(range||1)));const a=z<0?[24,45,76]:[18,45,49],b=z<0?[58,111,193]:[239,183,99],t=Math.pow(Math.abs(z),.75);return a.map((v,i)=>Math.round(v+(b[i]-v)*t))}
function render(){
 if(!datasets[scenario])return;
 const f=current(),n=f.u.length,canvas=$("field"),ctx=canvas.getContext("2d"),buffer=document.createElement("canvas");buffer.width=n;buffer.height=n;const bc=buffer.getContext("2d"),image=bc.createImageData(n,n);
 let values=f[observable];if(observable==="speed")values=f.u.map((row,i)=>row.map((v,j)=>Math.hypot(v,f.v[i][j])));
 const max=Math.max(1e-12,...values.flat().map(Math.abs));
 for(let x=0;x<n;x++)for(let y=0;y<n;y++){const rgb=observable==="speed"?color(values[x][y]-max/2,max/2):color(values[x][y],max);const offset=4*(y*n+x);image.data.set([...rgb,255],offset)}
 bc.putImageData(image,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=true;ctx.drawImage(buffer,0,0,canvas.width,canvas.height);
 if($("vectors").checked){const stride=4;ctx.strokeStyle="#e5fff3aa";ctx.lineWidth=1.15;for(let x=2;x<n;x+=stride)for(let y=2;y<n;y+=stride){const vx=f.u[x][y],vy=f.v[x][y],scale=13;const px=(x+.5)/n*canvas.width,py=(y+.5)/n*canvas.height,dx=vx*scale,dy=vy*scale;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px+dx,py+dy);ctx.stroke();ctx.beginPath();ctx.arc(px+dx,py+dy,1.6,0,2*Math.PI);ctx.fillStyle="#e5fff3";ctx.fill()}}
 $("fieldTitle").textContent=({vorticity:"Vorticity ω",pressure:"Pressure p",speed:"Speed |u|"}[observable]);$("low").textContent=observable==="speed"?"0":number(-max);$("high").textContent=number(max);
 $("time").textContent="t = "+f.time.toFixed(4)+" s";$("energy").textContent=number(f.energy);$("divergence").textContent=number(f.max_divergence);$("step").textContent=f.step.toString().padStart(4,"0");
 $("timeline").max=datasets[scenario].frames.length-1;$("timeline").value=frameIndex;
 $("explain").textContent=descriptions[stage]+" At this selected completed frame: kinetic energy = "+number(f.energy)+", max discrete divergence = "+number(f.max_divergence)+". Intermediate stage fields are not exported; this panel explains solver operations, not a recorded stage-by-stage field trace.";
}
function setScenario(name){scenario=name;frameIndex=0;document.querySelectorAll(".scenario").forEach(b=>b.classList.toggle("active",b.dataset.scenario===name));$("scenarioDesc").textContent=name==="shear"?"u = sin(2πy), v = 0. Analytic Fourier shear mode; pressure projection preserves divergence-free flow.":"u = sin(2πy), v = sin(2πx). Nonlinear interaction with pressure projection.";render()}
function setPlaying(value){playing=value;$("play").textContent=value?"Ⅱ Pause":"▶ Play";clearInterval(timer);if(value)timer=setInterval(()=>{frameIndex=(frameIndex+1)%datasets[scenario].frames.length;render()},110)}
document.querySelectorAll(".scenario").forEach(b=>b.addEventListener("click",()=>setScenario(b.dataset.scenario)));
document.querySelectorAll(".fieldbtn").forEach(b=>b.addEventListener("click",()=>{observable=b.dataset.field;document.querySelectorAll(".fieldbtn").forEach(c=>c.classList.toggle("active",c===b));render()}));
document.querySelectorAll(".stage").forEach(b=>b.addEventListener("click",()=>{stage=Number(b.dataset.stage);document.querySelectorAll(".stage").forEach(c=>c.classList.toggle("active",c===b));render()}));
$("vectors").addEventListener("change",render);$("timeline").addEventListener("input",e=>{frameIndex=Number(e.target.value);setPlaying(false);render()});$("play").addEventListener("click",()=>setPlaying(!playing));
Promise.all(["nonlinear","shear"].map(name=>fetch("./data/"+name+".json").then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(data=>{if(!Array.isArray(data.frames)||!data.frames.length)throw Error("Missing frames");datasets[name]=data}))).then(render).catch(e=>{$("explain").textContent="Simulation frames unavailable: "+e.message+". Check the GitHub Actions data build."});
