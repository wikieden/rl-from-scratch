// 随机游走 A–E（例 6.2）：从 C 出发，左出 0、右出 +1；返回 [[状态, 奖励, 下一个状态或 -1], ...]
const WALK_TRUE=[1,2,3,4,5].map(x=>x/6),WN=['A','B','C','D','E'];
function walkEpisode(){let s=2,ep=[];for(;;){const ns=s+(rnd()<0.5?1:-1);if(ns<0){ep.push([s,0,-1]);return ep}if(ns>4){ep.push([s,1,-1]);return ep}ep.push([s,0,ns]);s=ns}}
function rmsErr(V){return Math.sqrt(V.reduce((a,v,i)=>a+(v-WALK_TRUE[i])**2,0)/5)}
function walkRow(V,y,cur,hi){ // 画五个格子和价值柱
  for(let i=0;i<5;i++){const x=110+i*100;el('rect',{x,y,width:80,height:50,rx:8,fill:i===cur?'var(--accent-bg)':'var(--card)',stroke:hi&&hi.includes(i)?'var(--warm)':'var(--line)','stroke-width':hi&&hi.includes(i)?3:1},sv);
    txt(WN[i],{x:x+40,y:y+31,'text-anchor':'middle',class:'big'},sv)}
  txt('左出 0',{x:60,y:y+31,'text-anchor':'middle',class:'m'},sv);txt('右出 +1',{x:625,y:y+31,'text-anchor':'middle',class:'m'},sv);
  if(cur!==null&&cur>=0)el('circle',{cx:150+cur*100,cy:y-14,r:9,fill:'var(--warm)'},sv);
  const base=y+190,h=120;el('line',{x1:100,y1:base,x2:600,y2:base,stroke:'var(--line)'},sv);
  for(let i=0;i<5;i++){const x=125+i*100;el('rect',{x,y:base-V[i]*h,width:50,height:V[i]*h,fill:'var(--accent)',opacity:.75},sv);
    el('line',{x1:x-8,y1:base-WALK_TRUE[i]*h,x2:x+58,y2:base-WALK_TRUE[i]*h,stroke:'var(--warm)','stroke-dasharray':'5 4'},sv);
    txt(V[i].toFixed(3),{x:x+25,y:base+16,'text-anchor':'middle',class:'m'},sv)}}
