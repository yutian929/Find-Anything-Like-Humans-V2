const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'map_editor.html'), 'utf8');
const source = html.split('<script>')[1].split('</script>')[0];
const elements = new Map();
const fills = [];
const context = new Proxy({
    fillRect(left, top) { fills.push({left, top, color: this.fillStyle}); },
}, {get: (target, name) => target[name] ?? (() => {})});

function makeElement() {
    return {
        value: '', textContent: '', style: {}, dataset: {}, children: [], listeners: {},
        classList: {toggle() {}},
        addEventListener(name, callback) { this.listeners[name] = callback; },
        getContext() { return context; },
        getBoundingClientRect() { return {left: 0, top: 0, width: this.width, height: this.height}; },
        setPointerCapture() {},
        click() {},
    };
}
for (const match of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
    const item = makeElement();
    item.value = match[0].match(/\bvalue="([^"]*)"/)?.[1] ?? '';
    elements.set(match[1], item);
}
let exported;
const sandbox = {
    document: {
        getElementById: identifier => elements.get(identifier),
        createElement: () => makeElement(),
    },
    Blob: class {
        constructor(parts) { exported = JSON.parse(parts.join('')); }
    },
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}},
    setTimeout(callback) { callback(); },
};
const browserLocation = {href: 'http://127.0.0.1:8765/'};
Object.defineProperty(sandbox, 'location', {
    configurable: false,
    get() { return browserLocation; },
    set() { throw new Error('must not overwrite browser location'); },
});

async function test() {
    vm.runInNewContext(source, sandbox);
    assert.match(elements.get('status').textContent, /地图已创建/);
    assert.equal(sandbox.location, browserLocation);
    const canvas = elements.get('canvas');
    const pointer = (clientX, clientY) => ({button: 0, pointerId: 1, clientX, clientY});
    const click = (left = 150, top = 150) => {
        canvas.listeners.pointerdown(pointer(left, top));
        canvas.listeners.pointerup();
    };
    const useTool = tool => elements.get('tools').listeners.click({target: {dataset: {tool}}});
    const save = () => { elements.get('save').onclick(); return exported; };

    canvas.listeners.pointerdown(pointer(150, 150));
    canvas.listeners.pointermove(pointer(330, 150));
    canvas.listeners.pointerup();
    assert.equal(save().grid.rows[2].slice(2, 6), '####');
    assert.ok(fills.some(fill => fill.left === 120 && fill.top === 120 && fill.color === '#000000'));
    elements.get('undo').onclick();
    assert.equal(save().grid.rows[2].slice(2, 6), '....');

    useTool('object');
    elements.get('objectHeight').value = '0.85';
    click();
    elements.get('objectHeight').value = '1.5';
    elements.get('objectHeight').onchange();
    click(270, 150);
    assert.deepEqual(save().objects.map(obj => obj.position[2]), [0.85, 1.5]);

    useTool('select');
    click();
    elements.get('objectHeight').value = '2';
    elements.get('objectHeight').onchange();
    assert.deepEqual(save().objects.map(obj => obj.position[2]), [2, 1.5]);
    elements.get('undo').onclick();
    assert.deepEqual(save().objects.map(obj => obj.position[2]), [0.85, 1.5]);

    const boxBefore=save().objects[0].aabb;
    assert.ok(Math.abs(boxBefore.min[2]-.6)<1e-9);
    assert.ok(Math.abs(boxBefore.max[2]-1.1)<1e-9);
    useTool('select');click();
    elements.get('objectSizeX').value='1.2';elements.get('objectSizeX').onchange();
    const resized=save().objects[0];
    assert.ok(Math.abs(resized.aabb.max[0]-resized.aabb.min[0]-1.2)<1e-9);
    assert.match(elements.get('heightDetails').textContent,/底部 0.600 m.*顶部 1.100 m/);
    assert.match(elements.get('heightLegend').textContent,/顶部高度/);
    elements.get('objectSizeZ').value='3';elements.get('objectSizeZ').onchange();
    assert.match(elements.get('status').textContent,/不能低于地面/);
    assert.deepEqual(save().objects[0],resized);
    elements.get('objectSizeZ').value='.5';
    // Select through the footprint, beyond the former center-point hit radius.
    useTool('select');click(210,150);
    assert.match(elements.get('heightHint').textContent,/已选择 o1/);
    const saved = save();
    assert.deepEqual(Object.keys(saved).sort(), ['grid', 'name', 'objects']);
    assert.deepEqual(Object.keys(saved.objects[0]).sort(), ['aabb', 'id', 'position']);
    useTool('remove');
    click();
    assert.equal(save().objects.length, 1);
    const inputEvent = data => ({target: {files: [{text: async () => JSON.stringify(data)}], value: 'map.json'}});
    await elements.get('file').onchange(inputEvent(saved));
    assert.deepEqual(save(), saved);
    const invalid = structuredClone(saved);
    invalid.grid.resolution = -1;
    await elements.get('file').onchange(inputEvent(invalid));
    assert.match(elements.get('status').textContent, /打开失败/);
    assert.deepEqual(save(), saved);
    const badBox=structuredClone(saved);badBox.objects[0].aabb.max[0]=badBox.objects[0].aabb.min[0];
    await elements.get('file').onchange(inputEvent(badBox));
    assert.match(elements.get('status').textContent,/打开失败/);assert.deepEqual(save(),saved);
    const oldPoints=structuredClone(saved);oldPoints.objects.forEach(obj=>delete obj.aabb);
    await elements.get('file').onchange(inputEvent(oldPoints));
    assert.match(elements.get('status').textContent,/旧物体仅有中心点/);
    assert.ok(!save().objects[0].aabb);
    useTool('select');click();elements.get('applyBox').onclick();
    assert.ok(save().objects[0].aabb);assert.ok(!save().objects[1].aabb);
    const legacy = {...saved, camera: {}, robot: {}, planner: {}};
    await elements.get('file').onchange(inputEvent(legacy));
    assert.deepEqual(save(), saved);
    elements.get('columns').value = '300';
    elements.get('rows').value = '300';
    elements.get('resolution').value = '0.05';
    elements.get('new').onclick();
    assert.match(elements.get('status').textContent, /地图已创建/);
    assert.equal(save().grid.rows.length, 300);
    assert.ok(save().grid.rows.every(row => row.length === 300));
    assert.equal(save().grid.resolution, 0.05);
    assert.ok(canvas.width <= 960 && canvas.height <= 960);
    useTool('wall');
    click(canvas.width * 299.5 / 300, canvas.height * 299.5 / 300);
    assert.equal(save().grid.rows[299][299], '#');
    useTool('object');
    click(canvas.width / 2, canvas.height / 2);
    assert.deepEqual(save().objects[0].position.slice(0, 2), [7.5, 7.5]);
    const largeMap = save();
    await elements.get('file').onchange(inputEvent(largeMap));
    assert.match(elements.get('status').textContent, /地图已打开/);
    assert.deepEqual(save(), largeMap);
    let prevented = false;
    canvas.listeners.wheel({clientX: 240, clientY: 240, deltaY: -Math.log(2) / 0.002,
        deltaMode: 0, preventDefault() { prevented = true; }});
    assert.ok(prevented);
    assert.equal(elements.get('zoomLevel').textContent, '200%');
    assert.deepEqual(save(), largeMap);
    useTool('object');
    click(240, 240);
    assert.deepEqual(save().objects.at(-1).position.slice(0, 2), [3.75, 11.25]);
    elements.get('undo').onclick();
    canvas.listeners.pointerdown({button: 2, pointerId: 1, clientX: 240, clientY: 240, preventDefault() {}});
    canvas.listeners.pointermove(pointer(340, 290));
    canvas.listeners.pointerup();
    assert.deepEqual(save(), largeMap);
    click(340, 290);
    assert.deepEqual(save().objects.at(-1).position.slice(0, 2), [3.75, 11.25]);
    elements.get('undo').onclick();
    useTool('wall');
    click(343.2, 293.2);
    assert.equal(save().grid.rows[75][75], '#');
    elements.get('undo').onclick();
    elements.get('resetView').onclick();
    assert.equal(elements.get('zoomLevel').textContent, '100%');
    assert.deepEqual(save(), largeMap);
    elements.get('brushSize').value = '5';
    useTool('unknown');
    canvas.listeners.pointerdown(pointer(50.5 * 3.2, 50.5 * 3.2));
    canvas.listeners.pointermove(pointer(55.5 * 3.2, 50.5 * 3.2));
    canvas.listeners.pointerup();
    for (let row = 48; row <= 52; row++) assert.equal(save().grid.rows[row].slice(48, 58), '?'.repeat(10));
    elements.get('undo').onclick();
    assert.deepEqual(save(), largeMap);
    useTool('wall');
    click(0.5 * 3.2, 0.5 * 3.2);
    assert.equal(save().grid.rows[0].slice(0, 4), '###.');
    assert.ok(save().grid.rows.every(row => row.length === 300));
    elements.get('undo').onclick();
    elements.get('strokeMode').listeners.click({target: {dataset: {shape: 'rectangle'}}});
    canvas.listeners.pointerdown(pointer(20.5 * 3.2, 30.5 * 3.2));
    canvas.listeners.pointermove(pointer(10.5 * 3.2, 12.5 * 3.2));
    assert.deepEqual(save(), largeMap); // A preview must not mutate the grid.
    canvas.listeners.pointerup();
    for (let row = 12; row <= 30; row++) assert.equal(save().grid.rows[row].slice(10, 21), '#'.repeat(11));
    elements.get('undo').onclick();
    assert.deepEqual(save(), largeMap);
    useTool('erase');
    canvas.listeners.pointerdown(pointer(298.5 * 3.2, 298.5 * 3.2));
    canvas.listeners.pointermove(pointer(1200, 1200));
    canvas.listeners.pointerup();
    assert.equal(save().grid.rows[299][299], '.');
    assert.ok(save().grid.rows.every(row => row.length === 300));
    elements.get('undo').onclick();
    assert.deepEqual(save(), largeMap);
    useTool('wall');
    canvas.listeners.pointerdown(pointer(100, 100));
    canvas.listeners.pointermove(pointer(300, 300));
    canvas.listeners.pointercancel();
    assert.deepEqual(save(), largeMap);
    elements.get('columns').value = '513';
    elements.get('new').onclick();
    assert.match(elements.get('status').textContent, /4–512/);
    assert.deepEqual(save(), largeMap);
    const oversized = structuredClone(largeMap);
    oversized.grid.rows = Array(513).fill('.'.repeat(300));
    await elements.get('file').onchange(inputEvent(oversized));
    assert.match(elements.get('status').textContent, /打开失败/);
    assert.deepEqual(save(), largeMap);
    console.log('PASS: editor interactions, large maps, zoom/pan, wide brush, rectangle preview/fill, clipping, cancel, undo and save/load');
}
test().catch(error => { console.error(error); process.exitCode = 1; });
