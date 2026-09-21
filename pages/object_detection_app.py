import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# การตั้งค่าหน้าเว็บเบื้องต้น
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dog Breed Detection",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model/best.pt"
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# สีกรอบตรวจจับ (BGR) — โทนฟ้าน้ำเงินให้เข้ากับธีมเว็บ
BOX_COLOR = (216, 138, 61)   # ฟ้าอมน้ำเงินสด
LABEL_TEXT_COLOR = (11, 15, 25)

# ---------------------------------------------------------------------------
# โหลดโมเดล (แคชไว้ไม่ให้โหลดซ้ำทุกครั้งที่มีการโต้ตอบ)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล YOLO...")
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
class_names = model.model.names

def draw_detections(frame, results):
    """วาดกรอบ, track ID, ชื่อคลาส และ confidence score ลงบนเฟรม"""
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confs):
            x1, y1, x2, y2 = box
            label = f"#{track_id} {class_names[class_id]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), BOX_COLOR, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, LABEL_TEXT_COLOR, 2)
    return frame


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🐶 Dog Breed Detection")
    st.markdown("ระบบตรวจจับสายพันธุ์สุนัขด้วย **YOLO**")
    st.markdown("---")
    st.markdown("### สายพันธุ์ที่รองรับ")
    for cls_name in class_names.values():
        st.markdown(f"- {cls_name}")
    st.markdown("---")
    st.caption("โมเดล: `model/best.pt`")
    st.caption("ไฟล์อัปโหลดจะถูกเก็บไว้ที่โฟลเดอร์ `upload/`")


# ---------------------------------------------------------------------------
# ส่วนหัวหน้าเว็บ
# ---------------------------------------------------------------------------
st.title("🐶 ระบบตรวจจับสายพันธุ์สุนัขด้วย YOLO")
st.caption("ชิสุ  •  เฟรนช์บูลด็อก  •  โกลเด้นรีทรีฟเวอร์")
st.markdown("---")

tab_video, tab_webcam = st.tabs(["📹  อัปโหลดวิดีโอ", "🎥  เว็บแคมเรียลไทม์"])

# ---------------------------------------------------------------------------
# แท็บ 1: อัปโหลดวิดีโอ
# ---------------------------------------------------------------------------
with tab_video:
    st.subheader("อัปโหลดวิดีโอเพื่อตรวจจับสายพันธุ์สุนัข")

    col_upload, col_setting = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "เลือกไฟล์วิดีโอ (mp4, avi, mov, mkv)",
            type=["mp4", "avi", "mov", "mkv"],
        )

    with col_setting:
        conf_video = st.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05, key="conf_video")
        skip_frame = st.checkbox("ข้ามเฟรมคู่เพื่อความเร็ว (แนะนำ)", value=True)

    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"บันทึกไฟล์วิดีโอไว้ที่ `{save_path}` เรียบร้อยแล้ว")

        start_video = st.button("▶️  เริ่มตรวจจับ", key="start_video", use_container_width=True)

        if start_video:
            cap = cv2.VideoCapture(save_path)
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                count += 1
                if skip_frame and count % 2 != 0:
                    continue

                frame = cv2.resize(frame, (1020, 600))
                results = model.track(frame, persist=True, conf=conf_video)
                frame = draw_detections(frame, results)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                progress_bar.progress(min(count / total_frames, 1.0))
                status_text.caption(f"ประมวลผลเฟรมที่ {count} / {total_frames}")

            cap.release()
            status_text.empty()
            progress_bar.empty()
            st.info("✅ ตรวจจับวิดีโอเสร็จสิ้น")

# ---------------------------------------------------------------------------
# แท็บ 2: เว็บแคมเรียลไทม์
# ---------------------------------------------------------------------------
with tab_webcam:
    st.subheader("ตรวจจับสายพันธุ์สุนัขแบบเรียลไทม์จากเว็บแคม")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        conf_cam = st.slider("Confidence threshold ", 0.0, 1.0, 0.4, 0.05, key="conf_cam")
    with col_b:
        run_cam = st.checkbox("🔴 เปิดกล้องเว็บแคม")

    cam_placeholder = st.empty()

    st.caption(
        "หมายเหตุ: ฟีเจอร์นี้ต้องรันแอปบนเครื่องที่มีกล้องเว็บแคมต่ออยู่ "
        "หากต้องการหยุดกล้อง ให้ยกเลิกการติ๊กช่องด้านบนแล้วรีเฟรชหน้าเว็บ"
    )

    if run_cam:
        cap = cv2.VideoCapture(0)
        count = 0
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("ไม่พบเว็บแคม กรุณาตรวจสอบการเชื่อมต่อกล้อง")
                break

            count += 1
            if count % 2 != 0:
                continue

            frame = cv2.resize(frame, (1020, 600))
            results = model.track(frame, persist=True, conf=conf_cam)
            frame = draw_detections(frame, results)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
