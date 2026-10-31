const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../assets/live_map.html'), 'utf8');
let strokes=0, draws=0;
const context = {clearRect(){},fillRect(){},fillText(){},save(){},restore(){},translate(){},scale(){},
 drawImage(){draws++},strokeRect(){strokes++},beginPath(){},arc(){},fill(){},stroke(){},moveTo(){},lineTo(){},
 createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)}},putImageData(){}};
const elements=new Map();
function element(id){
 if(!elements.has(id))elements.set(id,{id,style:{},textContent:'',disabled:false,
  getBoundingClientRect(){return {width:900,height:600}},getContext(){return context},
  addEventListener(){},setPointerCapture(){}});
 return elements.get(id);
}
const state={
 fast:{status:'running',error:null,updated_at:Date.now()/1000},slow:{status:'idle',error:null,last_result:null},
 view:{grid:{rows:['....','.#..','....','....'],origin:[0,0],resolution:1},
 objects:[{id:'new',aabb:{min:[1,1,0],max:[2,2,1]}}],coverage:[[1,0,2]],revision:1,frame_id:'f1',
 map_id:'room',map_revision:'r1',added_ids:['new'],removed_ids:[],robot_position:[.5,.5],
 camera_position:[.5,.5],camera_forward:[1,0],rgb:'data:image/jpeg;base64,fixture'}};
const replies=[state],calls=[],timers=[];
const sandbox={console,Uint8ClampedArray,Set,Math,Date,
 document:{getElementById:element,createElement:()=>({getContext:()=>context})},
 window:{addEventListener(){}},setTimeout(fn){timers.push(fn)},setInterval(){},
 fetch:async(url,options)=>{calls.push([url,options]);const reply=replies.shift();if(reply instanceof Error)throw reply;
 return {ok:true,json:async()=>reply}}};
vm.createContext(sandbox);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],sandbox);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 await flush();
 assert.equal(element('replan').disabled,false);
 assert.equal(element('objects').textContent,1);
 assert(strokes>0 && draws>0,'map geometry should render');
 replies.push({accepted:true,message:'queued'});
 await element('replan').onclick();
 assert.equal(calls.at(-1)[0],'/api/replan');
 assert.equal(calls.at(-1)[1].method,'POST');
 assert.equal(element('replan').disabled,true);
 const newer=structuredClone(state);
 newer.view.revision=2;newer.slow.status='complete';
 newer.slow.last_result={source_map:{revision:1,map_id:'room',map_revision:'r1'},
 anchor_count:1,photo_count:2,anchors:[{position:[2,2]}],url:'/plans/0001/anchor_plan.html'};
 replies.push(newer);timers.shift()();await flush();
 assert.match(element('planText').textContent,/当前地图已更新/);
 assert.equal(element('planLink').href,'/plans/0001/anchor_plan.html');
 replies.push(new Error('offline'));timers.shift()();await flush();
 assert.equal(element('replan').disabled,true);
 assert.match(element('notice').textContent,/连接中断/);
 console.log('Live UI: render, replan POST, snapshot version and disconnect handling passed');
})().catch(error=>{console.error(error);process.exitCode=1});
