"""Capture one live RGB-D frame and visualize CuTR cuboids (no motion commands).

Run from repository root: python -m falhv2.tools.cubify_preview --help
"""
from __future__ import annotations

import argparse
import colorsys
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from ..model_clients import build_cubify_anything
from ..model_clients.cubify_anything import encode_result
from ..robot_adapters.anchor_map import AnchorMapRobot

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]


def color(index):
    return tuple(round(v*255) for v in colorsys.hsv_to_rgb((index*.61803398875)%1,.75,1))


def draw_cuboid(image, corners, intrinsics, rgb, label):
    h,w = image.shape[:2]
    def project(p):
        return (round(float(np.clip(intrinsics.fx*p[0]/p[2]+intrinsics.cx,-1e6,1e6))),
                round(float(np.clip(intrinsics.fy*p[1]/p[2]+intrinsics.cy,-1e6,1e6))))
    # Clip to the camera near plane, then to the image rectangle.
    for ia,ib in EDGES:
        a,b = corners[ia].copy(),corners[ib].copy()
        if a[2] < .05 and b[2] < .05:
            continue
        if a[2] < .05:
            a += (b-a)*((.05-a[2])/(b[2]-a[2]))
        if b[2] < .05:
            b += (a-b)*((.05-b[2])/(a[2]-b[2]))
        visible,p,q = cv2.clipLine((0,0,w,h),project(a),project(b))
        if visible:
            cv2.line(image,p,q,rgb[::-1],2,cv2.LINE_AA)
    front = corners[corners[:,2] >= .05]
    if len(front):
        points=np.array([project(p) for p in front])
        x=int(np.clip(points[:,0].min(),0,max(0,w-130)))
        y=int(np.clip(points[:,1].min(),22,h-5))
        cv2.rectangle(image,(x,y-19),(min(w-1,x+130),y+4),(20,20,20),-1)
        cv2.putText(image,label,(x+3,y),cv2.FONT_HERSHEY_SIMPLEX,.48,rgb[::-1],1,cv2.LINE_AA)


def save_preview(frame, result, prefix, *, threshold, elapsed_s, max_boxes):
    import plotly.graph_objects as go
    raw=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR)
    overlay=raw.copy()
    visible=result.cuboids[:max_boxes]
    for i,box in reversed(list(enumerate(visible))):
        draw_cuboid(overlay,box.corners_camera_m,frame.intrinsics,color(i),
                    f"#{i+1}  {box.confidence:.2f}")
    h,w=raw.shape[:2]
    canvas=np.full((h+94,w*2,3),24,np.uint8)
    canvas[64:64+h,:w]=raw
    canvas[64:64+h,w:]=overlay
    cv2.putText(canvas,"LIVE RGB",(16,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(240,240,240),2)
    cv2.putText(canvas,f"CUBIFY ANYTHING | {len(visible)}/{len(result.cuboids)} cuboids",
                (w+16,40),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),2)
    caption=f"Frame {frame.frame_id} | score >= {threshold:.2f} | inference {elapsed_s:.2f}s | boxes are predictions"
    cv2.putText(canvas,caption,(16,h+84),cv2.FONT_HERSHEY_SIMPLEX,.55,(210,210,210),1)
    for suffix,im in ((".png",canvas),("_rgb.png",raw)):
        if not cv2.imwrite(str(prefix)+suffix,im):
            raise OSError("Could not save preview image")
    np.savez_compressed(str(prefix)+"_rgbd.npz",rgb=frame.rgb,depth_m=frame.depth_m,
                        T_map_camera=frame.T_map_camera,T_map_base=frame.T_map_base)
    payload=encode_result(result)
    payload.update(intrinsics=asdict(frame.intrinsics),
                   T_map_camera=frame.T_map_camera.tolist(),T_map_base=frame.T_map_base.tolist(),
                   score_threshold=threshold,inference_s=elapsed_s,
                   displayed_boxes=len(visible))
    Path(str(prefix)+".json").write_text(json.dumps(payload,indent=2))

    # The measured point cloud and predictions share the optical camera frame.
    yy,xx=np.mgrid[0:h:5,0:w:5]
    z=frame.depth_m[::5,::5]
    mask=np.isfinite(z)&(z>.15)&(z<8.)
    k=frame.intrinsics
    xyz=np.stack(((xx-k.cx)*z/k.fx,(yy-k.cy)*z/k.fy,z),axis=-1)[mask]
    colors=frame.rgb[::5,::5][mask]
    traces=[go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode="markers",
                        marker=dict(size=1.5,color=[f"rgb({r},{g},{b})" for r,g,b in colors]),
                        name="Measured RGB-D",hoverinfo="skip")]
    for i,box in enumerate(visible):
        coords=[[],[],[]]
        for a,b in EDGES:
            for axis in range(3):
                coords[axis].extend([float(box.corners_camera_m[a,axis]),
                                    float(box.corners_camera_m[b,axis]),None])
        r,g,b=color(i)
        traces.append(go.Scatter3d(x=coords[0],y=coords[1],z=coords[2],mode="lines",
                                  line=dict(width=5,color=f"rgb({r},{g},{b})"),
                                  name=f"#{i+1} score={box.confidence:.3f}"))
    fig=go.Figure(traces)
    fig.update_layout(title=f"Cubify Anything | frame {frame.frame_id} | optical coordinates (meters)",
                      template="plotly_dark",margin=dict(l=0,r=0,b=0,t=50),
                      scene=dict(aspectmode="data",xaxis_title="X right",yaxis_title="Y down",
                                 zaxis_title="Z forward",
                                 camera=dict(up=dict(x=0,y=-1,z=0),eye=dict(x=1,y=-.7,z=-1.5))))
    fig.write_html(str(prefix)+".html",include_plotlyjs=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",default="/home/hri/models_weights/cubifyanything/cutr_rgbd.pth")
    parser.add_argument("--source-path",default="/home/hri/map_test_new_v8_clean")
    parser.add_argument("--server-url",help="Use an existing HTTP model server instead of local inference")
    parser.add_argument("--endpoint",default="ipc:///tmp/rgbd_pose.ipc")
    parser.add_argument("--threshold",type=float,default=.25)
    parser.add_argument("--max-boxes",type=int,default=20)
    parser.add_argument("--max-side",type=int,default=768)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--output-dir",default=str(Path(__file__).resolve().parents[1]/"outputs"))
    args=parser.parse_args(argv)
    if args.max_boxes < 1:
        parser.error("--max-boxes must be positive")
    config=dict(mode="remote",server_url=args.server_url) if args.server_url else dict(
        mode="local",path=args.weights,source_path=args.source_path,device=args.device,max_side=args.max_side)
    config["score_threshold"]=args.threshold
    client=build_cubify_anything(config)
    try:
        # This is a camera-frame preview, not a persistent map integration.
        with AnchorMapRobot(map_id="preview-only",map_revision="unverified",endpoint=args.endpoint) as robot:
            frame=robot.capture(timeout_s=5.)
        started=time.monotonic()
        result=client.detect(frame)
        elapsed=time.monotonic()-started
    finally:
        client.close()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    prefix=output/("cubify_"+datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"))
    save_preview(frame,result,prefix,threshold=args.threshold,elapsed_s=elapsed,max_boxes=args.max_boxes)
    print(json.dumps(dict(prefix=str(prefix),frame_id=frame.frame_id,timestamp_ns=frame.timestamp_ns,
                          predictions=len(result.cuboids),displayed=min(len(result.cuboids),args.max_boxes),
                          inference_s=elapsed,scores=[b.confidence for b in result.cuboids]),indent=2))


if __name__ == "__main__":
    main()
