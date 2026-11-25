"""Client for the Valen-Decision-1.5 /v1/decisions image protocol."""
import base64
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2


def bounded(value, maximum=1):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError('Valen returned an invalid numeric answer')
    return value


class Valen:
    def __init__(self, endpoint, model, timeout=120):
        self.endpoint, self.model, self.timeout = endpoint, model, timeout

    def decisions(self, bgr, questions):
        ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError('could not encode image for Valen')
        image = 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode()
        message = {'role': 'user', 'content': [{'type': 'input_image', 'image_url': image}]}
        payload = dict(model=self.model, input=[message], questions=questions)
        req = Request(self.endpoint, data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                body = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f'Valen HTTP {exc.code}: {exc.read(500).decode(errors="replace")}') from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f'Valen unavailable at {self.endpoint}: {exc}') from exc
        answers = {a['name']: a for a in body['answers']}
        for q in questions:
            if answers.get(q['name'], {}).get('type') != q['type']:
                raise ValueError(f"missing Valen answer for {q['name']}")
        return answers

    def inspect(self, image, target):
        levels = [{'label': str(i), 'description': d} for i, d in enumerate(
            ['Absent or unrecognizable', 'Partially visible', 'Clearly recognizable'])]
        a = self.decisions(image, [
            dict(name='present', type='predicate', instructions=f'Search request: {target}\nThe requested object is visible in the image.'),
            dict(name='visibility', type='score', levels=levels,
                 instructions=f'Search request: {target}\nHow clearly is the requested object visible?'),
            dict(name='region', type='bbox', image=0, instructions=f'Locate the object described in this request: {target}')])
        box = [bounded(n) for n in a['region']['bbox']]
        if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
            raise ValueError('Valen returned malformed xyxy bounds')
        return dict(present=bounded(a['present']['probability']),
                    visibility=bounded(a['visibility']['score'], 2),
                    bbox=box, bbox_confidence=bounded(a['region']['confidence']))

    def verify(self, crop, target):
        choices = [{'value': k, 'description': v} for k, v in [
            ('match', 'The requested object is clearly visible.'),
            ('other', 'It shows something else.'), ('uncertain', 'Too ambiguous to identify.')]]
        a = self.decisions(crop, [dict(name='identity', type='choice', choices=choices,
                                      instructions=f'Search request: {target}\nWhich answer describes this crop?')])['identity']
        if a['choice'] not in {c['value'] for c in choices}:
            raise ValueError('unknown Valen choice')
        return a['choice']
