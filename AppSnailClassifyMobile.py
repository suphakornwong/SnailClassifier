import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import pandas as pd
import torch
import streamlit.components.v1 as components

# Import ไฟล์ Function ที่เราแยกออกมา (ไม่ต้องแก้ไฟล์นี้แล้ว)
import DataFunction as df

def main():
    # แนะนำให้ใช้ layout="centered" สำหรับการแสดงผลบนมือถือ
    st.set_page_config(page_title="Image classification AI", layout="centered")
    
    # ==========================================
    # ส่วนหัวของแอปพลิเคชัน (ย้ายจาก Sidebar มาไว้หน้าหลัก)
    # ==========================================
    # ใช้ markdown เพื่อปรับขนาดตัวอักษรให้ดูซอฟต์ลง ไม่แย่งซีน Title หลัก
    st.markdown("<h5 style='text-align: center; color: white;'>งานวิจัยของศุภกร วงษ์เรืองพิบูล</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>โครงการของ ดร.สุนัดดา เชาวลิต</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>แผนวิจัยของ ดร.ยุวรินทร์ บุณทบ</h5>", unsafe_allow_html=True)
    st.markdown("---") # เพิ่มเส้นคั่นบางๆ ให้ดูเป็นระเบียบ
    st.title("🐌 Image classification snail pest")
    # ดึงปุ่มแนะนำระบบมาจาก DataFunction
    components.html(df.get_intro_button_html(), height=40)
    
    model_path = "best(1).pt"
    # ตอนนี้รองรับแค่โมเดลที่ชื่อว่า Best.pt
   
    # ==========================================
    # ส่วนอัปโหลดไฟล์ (ปรับปรุง UX สำหรับ Mobile)
    # ==========================================
    st.subheader("อัปโหลดหรือถ่ายรูปภาพศัตรูพืช")
    
    # สร้าง Tab ให้ผู้ใช้เลือกว่าจะอัปโหลดหรือถ่ายรูป
    tab1, tab2 = st.tabs(["📂 เลือกจากคลังภาพ", "📸 ถ่ายรูปใหม่"])
    
    uploaded_file = None
    
    with tab1:
        # ใส่ key="upload_gallery" เพื่อให้ระบบจำสถานะ widget
        file_from_gallery = st.file_uploader("เลือกไฟล์ภาพ...", type=["jpg", "jpeg", "png"], key="upload_gallery")
        if file_from_gallery is not None:
            uploaded_file = file_from_gallery
            
    with tab2:
        # เพิ่มกล้องถ่ายรูปโดยตรง ลดปัญหาเบราว์เซอร์รีเฟรช
        file_from_camera = st.camera_input("ถ่ายรูปศัตรูพืช", key="upload_camera")
        if file_from_camera is not None:
            uploaded_file = file_from_camera
            st.markdown("<h6 style='text-align: center; color: gray;'>หากต้องการกลับไปใช้รูปจาก gallery กรุณากดปุ่ม Clear photo ก่อนถึงจะอัพโหลดรูปได้</h5>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # บันทึกไฟล์ชั่วคราว
        # tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        # tfile.write(uploaded_file.read())
        
        img = Image.open(uploaded_file)
        # โหลดโมเดล YOLO
        model = YOLO(model_path)

        # อ่านไฟล์ภาพ
        # img = cv2.imread(tfile.name)

        # เริ่มการตรวจจับ
        # สามารถแสดง spinner โหลดข้อมูลเพื่อให้ฝั่ง Mobile รู้ว่าระบบกำลังประมวลผลอยู่
        with st.spinner('กำลังวิเคราะห์รูปภาพ...'):
                    results = model(img, conf=0.05, imgsz=640)[0]
        
        # ส่วนแสดงผลรูปภาพ
        img_with_boxes = results.plot()
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        
        # จุดสำคัญสำหรับ Mobile: ใช้ use_container_width=True เพื่อให้รูปพอดีจอมือถือ
        st.image(img_rgb, use_container_width=True)
        
        if results.probs is not None and hasattr(results, 'names'):
            # ดึง Top-5
            topk = torch.topk(results.probs.data, k=5)
            top_indices = topk.indices.tolist()
            top_scores = topk.values.tolist()

            top_classes = [results.names[i] for i in top_indices]
            top_confidences = [f"{score * 100:.2f}%" for score in top_scores]
            
            # ==========================================
            # เรียกใช้ Logic จาก DataFunction
            # ==========================================
            snail_info = df.process_snail_data(top_classes[0])
            
            # แสดงผลข้อความบนหน้าจอ (Subheaders)
            spoken_tokens = snail_info['tokens']
            
            st.markdown("---")
            st.subheader('📌 ผลการจำแนก')        
            
            # ป้องกัน Error กรณี List index out of range ถ้าชื่อสั้นกว่าปกติ
            display_name = spoken_tokens[0]
            sci_name_display = f"_{spoken_tokens[2]} {spoken_tokens[3]}_" if len(spoken_tokens) > 3 else ""
            family_display = spoken_tokens[5] if len(spoken_tokens) > 5 else "Unknown"
            
            # ใช้ st.success ให้กล่องข้อความดูโดดเด่นขึ้นบนมือถือ
            st.success(f"**ชื่อ:** {display_name} ({sci_name_display})  \n**วงศ์:** {family_display}  \n**ความมั่นใจ:** {top_confidences[0]}")
            
            # แสดงตาราง DataFrame
            df_result = pd.DataFrame({
                "อันดับ": [1, 2, 3, 4, 5], 
                "ชื่อชนิด": top_classes,
                "ความมั่นใจ": top_confidences
            })
            # บนมือถือตารางอาจจะดูยากนิดหน่อย แต่ Streamlit จัดการ scroll ซ้ายขวาให้
            st.dataframe(df_result, use_container_width=True)
            
            # ==========================================
            # สร้างปุ่มพูดและปุ่มป้องกันกำจัด (HTML/JS)
            # ==========================================
            st.markdown("---")
            tts_html = df.get_tts_html_script(snail_info, top_confidences[0], True)
            components.html(tts_html, height=80)
                        
        else:
            st.warning("ไม่สามารถดึงผลการจัดจำแนกจากโมเดลได้")
        
        # ลบไฟล์ชั่วคราว
        # try:
        #     tfile.close()
        #     os.unlink(tfile.name)
        # except Exception as e:
        #     st.error(f"เกิดข้อผิดพลาดในการลบไฟล์ชั่วคราว: {e}")
            
if __name__ == "__main__":

    main()







