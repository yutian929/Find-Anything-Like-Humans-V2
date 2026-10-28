// Exercise the standalone viewer with an in-memory DOM/canvas, no browser needed.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const template=fs.readFileSync(path.join(__dirname,'../assets/anchor_selector.html'),'utf8');
assert.ok(!template.includes('id="path"'));
const script=template.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
function fixture(){
 const a=(index)=>({region_id:1,position:[.5+index,.5],station_index:index,observations:[{yaw_rad:0,object_ids:['o1','o2']}]});
 const method=(n)=>({replay_enabled:true,title:n===1?'物体驱动':'位置驱动',anchors:Array.from({length:n},(_,i)=>a(i)),actions:Array.from({length:n},(_,i)=>({anchor_index:i,yaw_rad:0,object_ids:['o1','o2'],projected:[{id:'o1',u:40,v:20,depth_m:1,distance_m:1,qualified:true},{id:'o2',u:60,v:25,depth_m:2,distance_m:2,qualified:true}]})),candidate_count:3,votes:[[0,2],[1,2]],inverse:{o1:[0,1],o2:[0,1]},uncovered_object_ids:[],generation_seconds:.001,object_heatmaps:{o1:[[0,.9,0,900000],[1,.6,1,600000]],o2:[[0,.5,1,500000],[1,.8,0,800000]]},vote_trace:[{station_index:0,new_object_ids:['o1'],weighted_score:1.4},{station_index:1,new_object_ids:['o2'],weighted_score:.8}]});
 return {scene_name:'test map',grid:{rows:['....','....','....','....'],resolution:1,origin:[0,0]},objects:[{id:'o1',position:[1.5,1.5,.5]},{id:'o2',position:[2.5,2.5,.8]}],camera:{camera:{width:100,height:50}},methods:[method(2)],stations:[[0,.5,.5],[1,1.5,.5]],region_count:1,yaws_rad:[0,Math.PI/2]};
}
function start(data){
 let draws=0;
 const context=new Proxy({measureText:t=>({width:String(t).length*6})},{get:(obj,key)=>key in obj?obj[key]:(...args)=>{draws++;if(['arc','fillRect','lineTo','moveTo','clearRect'].includes(key))assert.ok(args.every(Number.isFinite),key+' must use finite coordinates');}});
 class Element{
  constructor(id){this.id=id;this.children=[];this.listeners={};this.style={};this.checked=['links','names'].includes(id);this.clientWidth=id==='projection'?300:800;this.clientHeight=id==='projection'?150:650;this.width=this.clientWidth;this.height=this.clientHeight;}
  getContext(){return context;} getBoundingClientRect(){return {width:800,height:650,left:0,top:0};}
  append(child){this.children.push(child);} replaceChildren(...children){this.children=children;}
  addEventListener(name,cb){this.listeners[name]=cb;} setPointerCapture(){} toDataURL(){return 'data:image/png;base64,TEST';} click(){}
  set innerHTML(value){throw Error('Unexpected HTML injection');}
 }
 const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
 get('data').textContent=JSON.stringify(data);get('map').parentElement=new Element('mapbox');
 const sandbox={document:{getElementById:get,createElement:type=>new Element(type)},window:{devicePixelRatio:1,addEventListener(){}},console};
 vm.createContext(sandbox);vm.runInContext(script,sandbox);
 return {get,sandbox,draws:()=>draws};
}
const {get,sandbox,draws}=start(fixture());
assert.equal(get('anchorsCount').textContent,2);assert.equal(get('photosCount').textContent,2);assert.equal(get('coverageCount').textContent,'2/2');
assert.equal(get('anchorsCount').textContent,2);assert.equal(get('anchorChoices').children.length,2);
assert.match(get('sceneInfo').textContent,/1 个安全连通区域/);
get('anchorChoices').children[1].onclick();assert.equal(get('detailTitle').textContent,'站位 A2');assert.match(get('details').textContent,/1\.500/);assert.match(get('details').textContent,/候选站位编号 1/);assert.ok(!get('details').textContent.includes('undefined'));
get('objectChoices').children[1].onclick();assert.equal(get('detailTitle').textContent,'物体 o2');assert.match(get('details').textContent,/0\.80/);assert.match(get('details').textContent,/A1、A2/);
get('photoChoices').children[0].onclick();assert.match(get('photoInfo').textContent,/2 个物体/);
get('heatmap').value='total';get('heatmap').onchange();
assert.equal(vm.runInContext('heatValues.get(0)[0]',sandbox),1.4);
assert.equal(vm.runInContext('heatValues.get(1)[0]',sandbox),1.4);
get('round').value=1;get('round').oninput();
assert.equal(vm.runInContext('heatValues.get(0)[0]',sandbox),.5);
assert.equal(vm.runInContext('heatValues.get(1)[0]',sandbox),.8);
assert.match(get('roundInfo').textContent,/剩余 1 个目标/);
get('heatmap').value='object:o1';get('heatmap').onchange();
assert.equal(get('detailTitle').textContent,'物体 o1');
assert.equal(vm.runInContext('heatValues.get(0)[0]',sandbox),.9); // Not the global count or sum.
assert.equal(vm.runInContext('heatValues.get(1)[0]',sandbox),.6);
assert.equal(get('round').disabled,true); // Single-object distributions are never removed by replay.
vm.runInContext('probeIndex=1;showHeatProbe()',sandbox);
assert.match(get('heatInfo').textContent,/得分 0.600000.*90°/);
get('heatmap').value='total';get('heatmap').onchange();get('round').value=2;get('round').oninput();
assert.equal(vm.runInContext('heatValues.size',sandbox),0);
assert.match(get('heatInfo').textContent,/没有合格观察权重/);
get('heatmap').value='none';get('heatmap').onchange();assert.match(get('heatTitle').textContent,/已关闭/);
get('heatmap').value='total';get('round').value=0;get('heatmap').onchange();
const before=vm.runInContext('zoom',sandbox);get('map').listeners.wheel({preventDefault(){},clientX:400,clientY:300,deltaY:-120});assert.ok(vm.runInContext('zoom',sandbox)>before);
get('reset').onclick();assert.equal(vm.runInContext('zoom',sandbox),1);
const target=vm.runInContext('world(1.5,1.5)',sandbox);get('map').listeners.pointerdown({clientX:target[0],clientY:target[1],pointerId:1});get('map').listeners.pointerup({clientX:target[0],clientY:target[1]});assert.equal(get('detailTitle').textContent,'物体 o1');
assert.ok(draws()>100);
const empty=fixture();empty.objects=[];empty.methods.forEach(m=>{m.anchors=[];m.actions=[];m.votes=[];m.inverse={};m.object_heatmaps={};m.vote_trace=[];});assert.equal(start(empty).get('coverageCount').textContent,'—');
const boxes=fixture();
boxes.observation_model='aabb_weighted_region';boxes.observation_settings={min_in_frame_fraction:.8,min_observation_score:.5,min_visible_pixels:400};
boxes.objects[0].aabb={min:[1,1,.2],max:[2,2,.8]};
boxes.methods[0].actions.forEach(a=>a.projected.forEach(p=>{p.metrics={in_frame_fraction:1,visible_fraction:.5,visible_pixels:250,projection_bounds:[10,5,70,45],reason:'observation_score',observation_score:.4,distance_weight:.8,occluder_ids:['o2']};p.qualified=false;}));
boxes.methods[0].uncovered_object_ids=['o1'];boxes.methods[0].actions.forEach(a=>a.object_ids=['o2']);boxes.methods[0].anchors.forEach(a=>a.observations[0].object_ids=['o2']);
boxes.object_diagnostics={o1:{metrics:boxes.methods[0].actions[0].projected[0].metrics}};
const boxUI=start(boxes);
assert.match(boxUI.get('modelInfo').textContent,/AABB/);
boxUI.get('objectChoices').children[0].onclick();
assert.match(boxUI.get('details').textContent,/底部 0.20 m.*顶部 0.80 m/);
assert.match(boxUI.get('details').textContent,/最佳已检查候选/);
assert.match(boxUI.get('qualityInfo').textContent,/未遮挡 50.0%.*遮挡物 o2/);
assert.match(boxUI.get('qualityInfo').textContent,/得分 0.400.*距离权重 0.800/);
assert.match(boxUI.get('photoInfo').textContent,/Box 投影外框/);
assert.equal(boxUI.get('photoChoices').children.length,2);
console.log('Anchor viewer: station/object/photo selection, drawing, zoom/reset, map picking and empty targets passed.');

// Exact solutions have no replay; switching restores a separate baseline trace.
const exact=fixture();exact.methods.unshift(structuredClone(exact.methods[0]));
exact.methods[0].replay_enabled=false;exact.methods[0].vote_trace=[];
exact.optimization={lexicographic_optimal:true,stage1:{proven_optimal:true,lower_bound:2},stage2:{proven_optimal:true,lower_bound:2,fixed_anchor_count:2}};
exact.baseline_cost=[3,4];exact.selected_cost=[2,2];exact.selection_method='lexicographic_milp';
const exactUI=start(exact);
assert.equal(exactUI.get('replayBar').hidden,true);
assert.equal(exactUI.get('round').disabled,true);
assert.match(exactUI.get('optimizationInfo').textContent,/3 站.*2 站.*[\s\S]*均已证明最优/);
assert.match(exactUI.get('heatTitle').textContent,/全部物体/);
exactUI.get('method').value=1;exactUI.get('method').onchange();
assert.equal(exactUI.get('replayBar').hidden,false);
exactUI.get('round').value=1;exactUI.get('round').oninput();
assert.equal(vm.runInContext('heatValues.get(0)[0]',exactUI.sandbox),.5);
exactUI.get('method').value=0;exactUI.get('method').onchange();
assert.equal(exactUI.get('round').value,0);
assert.equal(vm.runInContext('heatValues.get(0)[0]',exactUI.sandbox),1.4);
const limited=structuredClone(exact);limited.optimization.lexicographic_optimal=false;
limited.optimization.stage1.proven_optimal=false;limited.selection_method='weighted_greedy_fallback';
assert.match(start(limited).get('optimizationInfo').textContent,/尚未证明最优.*已验证基线/);
limited.optimization.fallback_reason='uncovered_targets';
assert.match(start(limited).get('optimizationInfo').textContent,/全覆盖不可行/);
const audited=structuredClone(exact);
audited.optimization.high_density_verification={repairs:[{}],final:{status:'passed',sample_grid:96,uncovered_object_ids:[]}};
assert.match(start(audited).get('optimizationInfo').textContent,/96×96 高密度复核通过；已重选 1 次/);
audited.optimization.high_density_verification.final={status:'failed',sample_grid:96,uncovered_object_ids:['o2']};
assert.match(start(audited).get('optimizationInfo').textContent,/高密度复核未通过：缺少 o2/);
console.log('Lexicographic viewer: proofs, fallback, static heatmap and separate baseline replay passed.');
