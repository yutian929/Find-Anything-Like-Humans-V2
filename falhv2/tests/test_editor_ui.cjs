const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'editor.html'), 'utf8');
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

    const saved = save();
    assert.deepEqual(Object.keys(saved).sort(), ['grid', 'name', 'objects']);
    assert.deepEqual(Object.keys(saved.objects[0]).sort(), ['id', 'position']);
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
    const legacy = {...saved, camera: {}, robot: {}, planner: {}};
    await elements.get('file').onchange(inputEvent(legacy));
    assert.deepEqual(save(), saved);
    console.log('PASS: startup, black obstacles, drag, undo, per-object height, selection, deletion and save/load');
}
test().catch(error => { console.error(error); process.exitCode = 1; });
