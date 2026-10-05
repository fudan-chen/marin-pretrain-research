(function(root){
  function windowsFor(n){
    if(!Number.isSafeInteger(n)||n<1)throw new Error('字符数必须是正整数');
    if(n<=2000)return [[0,n]];
    const m=Math.floor(n/2);return [[0,2000],[Math.max(0,m-1000),m+1000],[n-2000,n]];
  }
  function coverage(n){
    const spans=windowsFor(n), sorted=spans.slice().sort((a,b)=>a[0]-b[0]);let total=0,last=0;
    for(const [a,b] of sorted){total+=Math.max(0,b-Math.max(a,last));last=Math.max(last,b);}
    return {spans,covered:total,fraction:total/n};
  }
  const api={windowsFor,coverage};if(typeof module==='object'&&module.exports)module.exports=api;else root.QualityWindows=api;
})(typeof globalThis!=='undefined'?globalThis:this);
