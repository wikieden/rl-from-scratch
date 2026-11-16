// 《从零学强化学习》讲义动画共用的小环境（和 solutions/chapter-code/rl_envs.py 同一套数字）
const ROBOT={ // P[s][a] = [[概率, 下一个状态, 奖励], ...]；状态 0 = 高电、1 = 低电；动作 0 搜索、1 等待、2 充电
  S:['高电','低电'],A:['搜索','等待','充电'],
  P(beta){const al=0.8;return {0:{0:[[al,0,2],[1-al,1,2]],1:[[1,0,1]]},1:{0:[[beta,1,2],[1-beta,0,-3]],1:[[1,1,1]],2:[[1,0,0]]}}},
  PI:{0:{0:0.9,1:0.1},1:{0:0.2,1:0.3,2:0.5}}
};
function pick(probs){let u=rnd(),c=0;for(let i=0;i<probs.length;i++){c+=probs[i];if(u<c)return i}return probs.length-1}
// 网格：MOVES 顺序 上、下、右、左
const MOVES=[[-1,0],[1,0],[0,1],[0,-1]],ARW=['↑','↓','→','←'];
function grid5step(r,c,a){ // 例 3.5：返回 [nr, nc, 奖励]
  if(r===0&&c===1)return[4,1,10];if(r===0&&c===3)return[2,3,5];
  const nr=r+MOVES[a][0],nc=c+MOVES[a][1];if(nr<0||nr>4||nc<0||nc>4)return[r,c,-1];return[nr,nc,0]}
function grid5eval(g,useMax){ // 随机策略评估（useMax=false）或价值迭代（true）
  let v=Array(25).fill(0);for(let it=0;it<2000;it++){const nv=v.map((_,s)=>{const r=Math.floor(s/5),c=s%5;
    const qs=[0,1,2,3].map(a=>{const[nr,nc,rw]=grid5step(r,c,a);return rw+g*v[nr*5+nc]});
    return useMax?Math.max(...qs):qs.reduce((x,y)=>x+y,0)/4});
    let d=0;nv.forEach((x,i)=>d=Math.max(d,Math.abs(x-v[i])));v=nv;if(d<1e-9)break}return v}
function grid4step(s,a){ // 例 4.1：0 和 15 是终点，每步 −1
  if(s===0||s===15)return[s,0];const r=Math.floor(s/4),c=s%4,nr=r+MOVES[a][0],nc=c+MOVES[a][1];
  if(nr<0||nr>3||nc<0||nc>3)return[s,-1];return[nr*4+nc,-1]}
function grid4sweep(v,useMax){return v.map((_,s)=>{if(s===0||s===15)return 0;
  const qs=[0,1,2,3].map(a=>{const[ns,rw]=grid4step(s,a);return rw+v[ns]});return useMax?Math.max(...qs):qs.reduce((x,y)=>x+y,0)/4})}
function heat(v,lo,hi){const t=clamp((v-lo)/(hi-lo||1),0,1);return `rgba(24,95,165,${(0.08+0.72*t).toFixed(3)})`}
function fmt(x,d=1){const s=x.toFixed(d);return s==='-0.0'||s==='-0'?s.slice(1):s}
