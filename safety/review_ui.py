"""In-app annotation correction using a small Streamlit v2 canvas component."""

import base64
import json
from pathlib import Path

import streamlit as st

from safety.review import ReviewStore

_ZONE = st.components.v2.component("helmet_zone", html='''
<canvas width="320" height="180" aria-label="Draw helmet-required area"
style="width:100%;touch-action:none;background:#222"></canvas>
''', js='''
export default function({data,parentElement,setStateValue}) {
 const canvas=parentElement.querySelector('canvas'),ctx=canvas.getContext('2d');
 let start=null,zone=data.zone;
 function draw(){ctx.clearRect(0,0,320,180);ctx.strokeStyle='#ffd000';ctx.lineWidth=3;
  ctx.strokeRect(zone[0]*320,zone[1]*180,(zone[2]-zone[0])*320,(zone[3]-zone[1])*180);}
 function point(e){const r=canvas.getBoundingClientRect();return [
  Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),
  Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
 canvas.onpointerdown=e=>{start=point(e);canvas.setPointerCapture(e.pointerId);};
 canvas.onpointermove=e=>{if(!start)return;const end=point(e);
  zone=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];draw();};
 canvas.onpointerup=()=>{if(!start)return;start=null;
  if(zone[2]>zone[0]&&zone[3]>zone[1])setStateValue('zone',zone);};draw();
}
''')


def zone_editor():
    key = "drawn-helmet-zone"
    zone = st.session_state.get(key, {}).get("zone", [0., 0., 1., 1.])
    result = _ZONE(key=key, data={"zone": zone}, default={"zone": zone},
                   on_zone_change=lambda: None)
    return tuple(result.zone)

_EDITOR = st.components.v2.component("head_box_editor", html='''
<p>Drag empty space to add a box. Select a box to resize from its bottom-right corner.
Choose its class or delete it using the controls.</p>
<select aria-label="Box class"><option>helmet</option><option>bare_head</option></select>
<button type="button">Delete selected box</button>
<canvas tabindex="0" aria-label="Annotation canvas" style="width:100%;touch-action:none"></canvas>
<button type="button" data-save="true">Save review</button>
''', js='''
export default function({data, parentElement, setTriggerValue}) {
 const canvas = parentElement.querySelector('canvas'), ctx = canvas.getContext('2d');
 const select = parentElement.querySelector('select'), button = parentElement.querySelector('button');
 let boxes = canvas._draftBoxes ?? structuredClone(data.boxes), selected = Number(canvas.dataset.selected ?? -1), start = null, mode = null;
 if(selected >= boxes.length) selected = -1;
 const picture = new Image(); canvas.width = data.width; canvas.height = data.height;
 function draw() {
  canvas.dataset.selected = String(selected);
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(picture,0,0);
  boxes.forEach((b,i) => { const [x,y,r,bottom] = b.xyxy;
   ctx.strokeStyle = b.class === 'helmet' ? '#00ee55' : '#ff5555';
   ctx.lineWidth = i === selected ? 5 : 2; ctx.strokeRect(x,y,r-x,bottom-y);
   ctx.fillStyle = ctx.strokeStyle; ctx.font = '18px sans-serif';
   ctx.fillText(b.class,x,Math.max(18,y-4));
   if(i===selected) ctx.fillRect(r-6,bottom-6,12,12);
  });
 }
 function emit() { canvas._draftBoxes = boxes; }
 function point(e) { const r=canvas.getBoundingClientRect(); return [
  Math.max(0,Math.min(canvas.width,(e.clientX-r.left)*canvas.width/r.width)),
  Math.max(0,Math.min(canvas.height,(e.clientY-r.top)*canvas.height/r.height))]; }
 picture.onload=draw; picture.src=data.image;
 canvas.onpointerdown=e=>{
  const [x,y]=point(e); start=[x,y]; canvas.setPointerCapture(e.pointerId);
  if(selected>=0 && Math.abs(boxes[selected].xyxy[2]-x)<15 &&
     Math.abs(boxes[selected].xyxy[3]-y)<15) { mode='resize'; return; }
  selected=boxes.findLastIndex(b=>x>=b.xyxy[0]&&x<=b.xyxy[2]&&y>=b.xyxy[1]&&y<=b.xyxy[3]);
  if(selected>=0) { select.value=boxes[selected].class; mode='select'; }
  else { boxes.push({class:select.value,xyxy:[x,y,x,y]}); selected=boxes.length-1; mode='new'; }
  draw();
 };
 canvas.onpointermove=e=>{ if(!start || mode==='select')return;
  const [x,y]=point(e), b=boxes[selected];
  if(mode==='resize') { b.xyxy[2]=Math.max(b.xyxy[0]+1,x); b.xyxy[3]=Math.max(b.xyxy[1]+1,y); }
  else b.xyxy=[Math.min(start[0],x),Math.min(start[1],y),Math.max(start[0],x),Math.max(start[1],y)];
  draw();
 };
 canvas.onpointerup=()=>{ if(!start)return; start=null;
  if(mode==='select') { draw(); return; }
  boxes=boxes.filter(b=>b.xyxy[2]>b.xyxy[0]&&b.xyxy[3]>b.xyxy[1]); draw(); emit(); };
 select.onchange=()=>{ if(selected>=0) { boxes[selected].class=select.value; draw(); emit(); } };
 button.onclick=()=>{ if(selected>=0){boxes.splice(selected,1);selected=-1;draw();emit();} };
 parentElement.querySelector('[data-save]').onclick=()=>setTriggerValue('submitted',{boxes:structuredClone(boxes)});
}
''')


def review_page(detector):
    store = ReviewStore()
    st.subheader("Review camera annotations")
    st.info("Draft boxes are predictions. Check every visible head, including missed heads, before approval.")
    sources = store.sources()
    if not sources:
        st.caption("Prepare the review pack with: uv run review_data.py extract")
        return
    with st.expander("Source groups and dataset export"):
        st.caption("Put clips from the same site/session and confirmed duplicates in one group.")
        for source in sources:
            st.write(Path(source["path"]).name)
            group = st.text_input("Source group", source["group_id"], key=f"group-{source['hash']}")
            choices = [None, "train", "val", "test"]
            split = st.selectbox("Split", choices, index=choices.index(source["split"]),
                format_func=lambda v: v or "Unassigned", key=f"split-{source['hash']}")
            if st.button("Save source assignment", key=source["hash"]):
                store.assign(source["hash"], group, split)
                st.success("Assignment saved")
        version = st.text_input("New dataset version", "camera-v1")
        if st.button("Export approved annotations"):
            try:
                if Path(version).name != version or version in (".", "..", ""):
                    raise ValueError("Use a single directory name")
                st.success(str(store.export(store.root / "versions" / version)))
            except ValueError as exc:
                st.error(str(exc))
    source_map = {s["hash"]: s for s in sources}
    source_id = st.selectbox("Video source", list(source_map),
                            format_func=lambda key: Path(source_map[key]["path"]).name)
    source = source_map[source_id]
    with st.expander("Review video event intervals"):
        import cv2

        st.video(source["path"])
        st.caption("Mark each bare-head episode in a helmet-required area. Use normalized head boxes "
                   "(0–1). Split moving heads into intervals with a stable location. Review the entire clip.")
        rows = [{"start": r["start"], "end": r["end"], "box": r["box"]}
                for r in store.intervals(source["hash"])]
        value = st.text_area("Intervals JSON", json.dumps(rows, indent=2), key=f"intervals-{source['hash']}")
        st.caption('Example: [{"start": 1.0, "end": 3.5, "box": [0.2, 0.1, 0.3, 0.25]}]')
        if st.button("Save reviewed intervals"):
            capture = cv2.VideoCapture(source["path"])
            try:
                duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
                store.save_intervals(source["hash"], json.loads(value), duration)
                st.success("Reviewed intervals saved")
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                st.error(str(exc))
            finally:
                capture.release()
    timestamp = st.number_input("Additional frame timestamp (seconds)", min_value=0., step=.1)
    if st.button("Extract additional frame"):
        try:
            store.extract(source["path"], detector, [timestamp])
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    frames = [f for f in store.frames() if f["source"] == source["hash"]]
    st.caption(f"{sum(f['status'] == 'approved' for f in frames)} of {len(frames)} frames approved")
    if not frames:
        return
    frame_map = {f["id"]: f for f in frames}
    frame_id = st.selectbox("Frame", list(frame_map),
        format_func=lambda key: f"{frame_map[key]['timestamp']:.2f}s — {frame_map[key]['status']}")
    frame = frame_map[frame_id]
    key = f"annotation-{frame['id']}-{frame['revision']}"
    status = st.selectbox("Review decision", ["unreviewed", "approved", "excluded"],
        index=["unreviewed", "approved", "excluded"].index(frame["status"]), key=f"status-{key}")
    boxes = frame["boxes"]
    encoded = base64.b64encode(Path(frame["image"]).read_bytes()).decode()
    result = _EDITOR(key=key, data={"boxes": boxes, "width": frame["width"],
        "height": frame["height"], "image": "data:image/jpeg;base64," + encoded},
        on_submitted_change=lambda: None)
    st.caption("An approved empty frame becomes a background negative. Exclude uncertain frames.")
    if result.submitted is not None:
        try:
            store.save(frame, result.submitted["boxes"], status)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
