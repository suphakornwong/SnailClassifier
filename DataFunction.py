import json

# ==========================================
# ส่วนเก็บข้อมูล Text ยาวๆ (Snail Knowledge Base)
# ==========================================

INTRO_TEXT = "  สวัสดีครับ ผมเป็นปัญญาประดิษฐ์ ที่ชำนาญด้านการจัดจำแนกหอยศัตรูพืช กรุณาอัพโหลดรูปภาพศัตรูพืชที่คุณสงสัย ผมจะช่วยจัดจำแนกให้"

# ข้อความการป้องกันและกำจัด
INFO_CHERRY = """  หอยชนิดนี้เป็นศัตรูที่สำคัญของ ข้าว จำนวนประชากรที่เป็นระดับตัดสินใจให้ใช้สารเคมี คือมากกว่า 2 ตัวต่อตารางเมตรขึ้นไปในนาข้าว สำหรับรายชื่อสารกำจัดศัตรูพืช ขอแนะนำให้ใช้สารดังต่อไปนี้ 
หนึ่ง ใช้สาร นิโคลซาไมด์ โอลามีน หรือนิโคลซาไมด์ เอทาโนลามีน 83.1 เปอร์เซ็นต์ ดับบิวพี อัตราการใช้ 50 กรัมต่อไร่ ผสมน้ำพ่นให้ทั่วนาข้าวเน้นบริเวณที่เป็นแอ่งน้ำ หรือที่มีหอยมาก
สอง ใช้สาร เมทัลดีไฮด์แบบเหยื่อพิษ 5 เปอร์เซ็นต์ จีบี อัตราการใช้ 500 กรัมต่อไร่ โดยหว่านลงน้ำให้ทั่วในนาข้าว เน้นบริเวณที่เป็นแอ่งน้ำ หรือมีหอยมาก
หรือสาม ใช้ ผงกากเมล็ดชา ซึ่งจะมีสารสำคัญคือซาโปนิน ความเข้มข้นร้อยละ 10  อัตราการใช้ 3 กิโลกรัมต่อไร่ โดยหว่านลงน้ำให้ทั่วในนาข้าว
หมายเหตุ ขณะใช้สารกำจัดหอยทั้งสามชนิด ควรรักษาระดับน้ำในนาข้าวไว้ที่ 5 เซนติเมตร เป็นเวลาอย่างน้อย 3 วัน จึงจะได้ผลดีที่สุด"""

INFO_AMBER_SUBULI_MARTEN = """  หอยชนิดนี้เป็นศัตรูที่สำคัญของ ผักตระกูลกะหล่ำ และกล้วยไม้ จำนวนประชากรที่เป็นระดับตัดสินใจให้ใช้สารเคมี คือเจอหอยศัตรูพืชบนกล้วยไม้หรือกะหล่ำมากกว่า 8 ต้นขึ้นไป จากการสุ่มเช็คพืช 40 ต้น สำหรับรายชื่อสารกำจัดศัตรูพืช ขอแนะนำให้ใช้สารดังต่อไปนี้ 
หนึ่ง ใช้สาร นิโคลซาไมด์ โอลามีน 83.1 เปอร์เซ็นต์ ดับบิวพี อัตราการใช้ 40 กรัมต่อน้ำ 20 ลิตร โดยผสมน้ำและพ่นให้ถูกตัวหอยศัตรูพืช 
สอง ใช้สาร เมทัลดีไฮด์แบบเหยื่อพิษ 5 เปอร์เซ็นต์ จีบี อัตราการใช้ 1000 กรัมต่อไร่ ใช้เหยื่อพิษหว่านบนพื้นดินตามทางเดินระหว่างกล้วยไม้หรือผัก หรือหว่านบนวัสดุปลูกกล้วยไม้
หรือสาม ใช้ ผงกากเมล็ดชา ซึ่งจะมีสารสำคัญคือซาโปนิน ความเข้มข้นร้อยละ 10  โดยนำผงกากชามาต้มกับน้ำจนเดือด นำน้ำที่กรองได้มาพ่นให้ถูกตัวหอยศัตรูพืช ที่อยู่บนพื้นดินตามทางเดิน
อัตราการใช้ 1000 กรัมต่อน้ำ 20 ลิตร หรือ ใช้การหว่านกากชาที่อัตรา 5000 กรัมต่อไร่
หมายเหตุ ก่อนใช้สารกำจัดหอยทั้งสามชนิด จำเป็นต้องพ่นน้ำเปล่าให้ทั่วแปลง เพื่อชักนำให้หอยศัตรูพืช ออกมาจากที่ซ่อน ควรพ่นสารกำจัดหอยตอนเช้าตรู่ และในกล้วยไม้ หลีกเลี่ยงการพ่นสารไม่ให้ถูกส่วนของดอกกล้วยไม้โดยตรง"""

INFO_JUMPING = """  หอยชนิดนี้เป็นศัตรูที่สำคัญของ กล้วยไม้ จำนวนประชากรที่เป็นระดับตัดสินใจให้ใช้สารเคมี คือเจอหอยศัตรูพืชบนกล้วยไม้มากกว่า 8 ต้นขึ้นไป จากการสุ่มเช็คพืช 40 ต้น สำหรับรายชื่อสารกำจัดศัตรูพืช ขอแนะนำให้ใช้สารดังต่อไปนี้ 
หนึ่ง ใช้สาร นิโคลซาไมด์ โอลามีน 83.1 เปอร์เซ็นต์ ดับบิวพี อัตราการใช้ 40 กรัมต่อน้ำ 20 ลิตร โดยผสมน้ำและพ่นให้ถูกตัวหอยศัตรูพืช
สอง ใช้สาร เมทัลดีไฮด์แบบเหยื่อพิษ 5 เปอร์เซ็นต์ จีบี อัตราการใช้ 1000 กรัมต่อไร่ ใช้เหยื่อพิษหว่านบนพื้นดินตามทางเดินระหว่างกล้วยไม้หรือผัก หรือหว่านบนวัสดุปลูกกล้วยไม้
หรือสาม ใช้ ผงกากเมล็ดชา ซึ่งจะมีสารสำคัญคือซาโปนิน ความเข้มข้นร้อยละ 10  โดยนำผงกากชามาต้มกับน้ำจนเดือด นำน้ำที่กรองได้มาพ่นให้ถูกตัวหอยศัตรูพืช ที่อยู่บนพื้นดินตามทางเดิน
อัตราการใช้ 1000 กรัมต่อน้ำ 20 ลิตร หรือ ใช้การหว่านกากชาที่อัตรา 5000 กรัมต่อไร่
หมายเหตุ ก่อนใช้สารกำจัดหอยทั้งสามชนิด จำเป็นต้องพ่นน้ำเปล่าให้ทั่วแปลง เพื่อชักนำให้หอยศัตรูพืช ออกมาจากที่ซ่อน ควรพ่นสารกำจัดหอยตอนเช้าตรู่ และในกล้วยไม้ หลีกเลี่ยงการพ่นสารไม่ให้ถูกส่วนของดอกกล้วยไม้โดยตรง"""

INFO_SARIKA = """  หอยชนิดนี้เป็นศัตรูที่สำคัญของ ผักตระกูลกะหล่ำ กล้วยไม้ หม่อน และไม้ผลต่าง ๆ การป้องกันและกำจัด ขอแนะนำให้ใช้สารดังต่อไปนี้ 
หนึ่ง ใช้สาร นิโคลซาไมด์ โอลามีน 83.1 เปอร์เซ็นต์ ดับบิวพี อัตราการใช้ 40 กรัมต่อน้ำ 20 ลิตร โดยผสมน้ำและพ่นให้ถูกตัวหอยศัตรูพืช
สอง ใช้สาร เมทัลดีไฮด์แบบเหยื่อพิษ 5 เปอร์เซ็นต์ จีบี อัตราการใช้ 1000 กรัมต่อไร่ ใช้เหยื่อพิษหว่านบนพื้นดินตามทางเดินระหว่างกล้วยไม้หรือผัก หรือหว่านบนวัสดุปลูกกล้วยไม้
หรือสาม ใช้ ผงกากเมล็ดชา ซึ่งจะมีสารสำคัญคือซาโปนิน ความเข้มข้นร้อยละ 10  โดยนำผงกากชามาต้มกับน้ำจนเดือด นำน้ำที่กรองได้มาพ่นให้ถูกตัวหอยศัตรูพืช ที่อยู่บนพื้นดินตามทางเดิน
อัตราการใช้ 1000 กรัมต่อน้ำ 20 ลิตร หรือ ใช้การหว่านกากชาที่อัตรา 5000 กรัมต่อไร่
หมายเหตุ ก่อนใช้สารกำจัดหอยทั้งสามชนิด จำเป็นต้องพ่นน้ำเปล่าให้ทั่วแปลง เพื่อชักนำให้หอยศัตรูพืช ออกมาจากที่ซ่อน ควรพ่นสารกำจัดหอยตอนเช้าตรู่ และในกล้วยไม้ หลีกเลี่ยงการพ่นสารไม่ให้ถูกส่วนของดอกกล้วยไม้โดยตรง"""

INFO_LISSA = """  หอยชนิดนี้เป็นศัตรูที่สำคัญของ  กล้วย  กล้วยไม้  พริก  กาแฟ  กะหล่ำ  ทุเรียน  มะเขือยาว  ลองกอง  หม่อน  เห็ด  มะละกอ  ส้มโอ  ชมพู่  ส้มเขียวหวาน  การป้องกันและกำจัด ขอแนะนำให้ใช้สารดังต่อไปนี้ 
หนึ่ง ใช้สาร นิโคลซาไมด์ โอลามีน 83.1 เปอร์เซ็นต์ ดับบิวพี อัตราการใช้ 40 กรัมต่อน้ำ 20 ลิตร โดยผสมน้ำและพ่นให้ถูกตัวหอยศัตรูพืช
สอง ใช้สาร เมทัลดีไฮด์แบบเหยื่อพิษ 5 เปอร์เซ็นต์ จีบี อัตราการใช้ 1000 กรัมต่อไร่ ใช้เหยื่อพิษหว่านบนพื้นดินตามทางเดินระหว่างกล้วยไม้ ผักหรือ ต้นไม้ หรือหว่านบนวัสดุปลูกกล้วยไม้
หรือสาม ใช้ ผงกากเมล็ดชา ซึ่งจะมีสารสำคัญคือซาโปนิน ความเข้มข้นร้อยละ 10  โดยนำผงกากชามาต้มกับน้ำจนเดือด นำน้ำที่กรองได้มาพ่นให้ถูกตัวหอยศัตรูพืช ที่อยู่บนพื้นดินตามทางเดิน
อัตราการใช้ 1000 กรัมต่อน้ำ 20 ลิตร หรือ ใช้การหว่านกากชาที่อัตรา 5000 กรัมต่อไร่
หมายเหตุ ก่อนใช้สารกำจัดหอยทั้งสามชนิด จำเป็นต้องพ่นน้ำเปล่าให้ทั่วแปลง เพื่อชักนำให้หอยศัตรูพืช ออกมาจากที่ซ่อน ควรพ่นสารกำจัดหอยตอนเช้าตรู่ และในกล้วยไม้ หลีกเลี่ยงการพ่นสารไม่ให้ถูกส่วนของดอกกล้วยไม้โดยตรง"""

INFO_BRADYBAENA = """  หอยชนิดนี้มีรายงานจากต่างประเทศว่าหอยดังกล่าวเป็นศัตรูพืชของไม้ผลและผักหลายชนิด 
แต่ขออภัยด้วย หอยชนิดนี้ยังไม่มีคำแนะนำอย่างเป็นทางการจากกรมวิชาการเกษตร ในการป้องกันและกำจัดครับ"""

INFO_PILA = """  มีรายงานว่าหอยสกุลดังกล่าวเป็นศัตรูพืชของต้นเผือกในหมู่เกาะฮาวาย สหรัฐอเมริกา 
แต่ขออภัยด้วย หอยชนิดนี้ยังไม่มีคำแนะนำอย่างเป็นทางการจากกรมวิชาการเกษตร ในการป้องกันและกำจัดครับ"""

INFO_OTHER_WATER_SNAIL = """  หอยชนิดนี้เป็นศัตรูที่สำคัญของ บัว และ ไม้น้ำสวยงาม 
แต่ขออภัยด้วย หอยชนิดนี้ยังไม่มีคำแนะนำอย่างเป็นทางการจากกรมวิชาการเกษตร ในการป้องกันและกำจัดครับ"""

INFO_NULL = """  ภาพดังกล่าวไม่ใช่หอยศัตรูพืชครับ จึงไม่สามารถจัดจำแนกได้"""


# ==========================================
# Function Helper สำหรับสร้าง HTML
# ==========================================

def get_intro_button_html():
    """คืนค่า HTML String สำหรับปุ่มแนะนำระบบ"""
    return f"""
    <script>
    function speakIntro() {{
        const synth = window.speechSynthesis; 
        const msg = new SpeechSynthesisUtterance({json.dumps(INTRO_TEXT)}); 
        msg.lang = "th-TH"; 
        msg.pitch = 1.0; 
        msg.rate = 0.7; 

        function setVoiceAndSpeak() {{ 
            const voices = synth.getVoices(); 
            const femaleVoice = voices.find(v => v.lang === 'th-TH' && /female|หญิง|wom/i.test(v.name)); 
            const thVoice = femaleVoice || voices.find(v => v.lang === 'th-TH'); 
            if (thVoice) {{ 
                msg.voice = thVoice;
            }} 
            synth.cancel(); 
            synth.speak(msg); 
        }} 

        if (synth.getVoices().length === 0) {{ 
            synth.onvoiceschanged = setVoiceAndSpeak; 
        }} else {{ 
            setVoiceAndSpeak(); 
        }} 
    }}
    </script>
    <button onclick="speakIntro()">🔊 แนะนำระบบ</button>
    """

def process_snail_data(top_class_name):
    """
    รับค่าชื่อ Class จาก YOLO (เช่น 'Parmarion_martensi')
    คืนค่าเป็น Dictionary ที่ประกอบด้วยข้อมูลสำหรับแสดงผลและ TTS
    """
    # แยกคำด้วยช่องว่าง
    tokens = top_class_name.split()
    
    # ค่าเริ่มต้น
    speech_protect = ""
    display_info = tokens # Default list
    
    first_token = tokens[0]

    # Handle Null case explicitly for first token check
    if first_token == "Null":
        first_token = "Null_Class"
    
    if first_token == "Parmarion_martensi":
        display_info = [" ทากเล็บมือนาง", " มีชื่อวิทยาศาสตร์คือ", " Parmarion", " martensi", " จัดอยู่ในวงศ์", " Ariophantidae"]
        speech_protect = INFO_AMBER_SUBULI_MARTEN
        
    elif first_token == "Subulina_octona":
        display_info = [" หอยข้าวสารยอดมน", " มีชื่อวิทยาศาสตร์คือ", " Subulina", " octona", " จัดอยู่ในวงศ์", " Subulinidae"]
        speech_protect = INFO_AMBER_SUBULI_MARTEN
        
    elif first_token == "Pila_sp":
        display_info = [" หอยโข่ง", " มีชื่อวิทยาศาสตร์ว่า", " Pila", " sp.", " จัดอยู่ในวงศ์", " Ampullariidae"]
        speech_protect = INFO_PILA
        
    elif first_token == "Physella_acuta":
        display_info = [" หอยคันหอยบัวหรือหอยบ่อ", " มีชื่อวิทยาศาสตร์ว่า", " Physella", " acuta", " จัดอยู่ในวงศ์", " Physidae"]
        speech_protect = INFO_OTHER_WATER_SNAIL
        
    elif first_token == "Radix_sp":
        display_info = [" หอยคันหอยบัวหรือหอยบ่อ", " มีชื่อวิทยาศาสตร์ว่า", " Radix", " sp.", " จัดอยู่ในวงศ์", " Lymnaeidae"]
        speech_protect = INFO_OTHER_WATER_SNAIL
        
    elif first_token == "Indoplanorbis_exustus":
        display_info = [" หอยคันหรือหอยคันอินโด", " มีชื่อวิทยาศาสตร์ว่า", " Indoplanorbis", " exustus", " จัดอยู่ในวงศ์", " Bulinidae"]
        speech_protect = INFO_OTHER_WATER_SNAIL
        
    elif first_token == "Melanoides_tuberculata":
        display_info = [" หอยเจดีย์ลายเสือ", " มีชื่อวิทยาศาสตร์ว่า", " Melanoides", " tuberculata", " จัดอยู่ในวงศ์", " Thiaridae"]
        speech_protect = INFO_OTHER_WATER_SNAIL
        
    elif first_token == "Allopeas_gracile":
        display_info = [" หอยเจดีย์เล็กหรือหอยข้าวสารธรรมดา", " มีชื่อวิทยาศาสตร์คือ", " Allopeas", " gracile", " จัดอยู่ในวงศ์", " Subulinidae"]
        speech_protect = INFO_AMBER_SUBULI_MARTEN
        
    elif first_token == "Paropeas_sp":
        display_info = [" หอยเจดีย์ใหญ่", " มีชื่อวิทยาศาสตร์คือ", " Paropeas", " sp.", " จัดอยู่ในวงศ์", " Subulinidae"]
        speech_protect = INFO_AMBER_SUBULI_MARTEN
        
    elif first_token == "Pomacea_canaliculata":
        display_info = [" หอยเชอรี่", " มีชื่อวิทยาศาสตร์คือ", " Pomacea", " canaliculata", " จัดอยู่ในวงศ์", " Ampullariidae"]
        speech_protect = INFO_CHERRY
        
    elif first_token == "Sarika_siamensis":
        display_info = [" หอยดักดานหรือหอยทากสยาม", " มีชื่อวิทยาศาสตร์คือ", " Sarika", " siamensis", " จัดอยู่ในวงศ์", " Ariophantidae"]
        speech_protect = INFO_SARIKA
        
    elif first_token == "Lissachatina_fulica":
        display_info = [" หอยทากยักษ์แอฟริกา", " มีชื่อวิทยาศาสตร์คือ", " Lissachatina", " fulica", " จัดอยู่ในวงศ์", " Achatinidae"]
        speech_protect = INFO_LISSA
        
    elif first_token == "Ovachlamys_fulgens":
        display_info = [" หอยเลขหนึ่ง", " มีชื่อวิทยาศาสตร์คือ", " Ovachlamys", " fulgens", " จัดอยู่ในวงศ์", " Helicarionidae"]
        speech_protect = INFO_JUMPING
        
    elif first_token == "Bradybaena_similaris":
        display_info = [" หอยสะดือบุ๋ม", " มีชื่อวิทยาศาสตร์ว่า", " Bradybaena", " similaris", " จัดอยู่ในวงศ์", " Camaenidae"]
        speech_protect = INFO_BRADYBAENA
        
    elif first_token == "Sarika_resplendens":
        display_info = [" หอยสาริกา", " มีชื่อวิทยาศาสตร์คือ", " Sarika", " resplendens", " จัดอยู่ในวงศ์", " Ariophantidae"]
        speech_protect = INFO_SARIKA
        
    elif first_token == "Succinea_sp":
        display_info = [" หอยอำพัน", " มีชื่อวิทยาศาสตร์คือ", " Succinea", " sp.", " จัดอยู่ในวงศ์", " Succineidae"]
        speech_protect = INFO_AMBER_SUBULI_MARTEN
        
    elif first_token == "Pomacea_maculata":
        display_info = [" หอยแอปเปิ้ลเกาะ", " มีชื่อวิทยาศาสตร์คือ", " Pomacea", " maculata", " จัดอยู่ในวงศ์", " Ampullariidae"]
        speech_protect = INFO_CHERRY

    elif first_token == "Null_Class":
        # แก้ไขข้อความตรงนี้ให้สั้นลง เพื่อให้ Logic ตรวจสอบง่ายขึ้น
        display_info = ["ไม่ใช่หอยศัตรูพืช", "ไม่มีข้อมูล", " ", "ไม่มีข้อมูล", " ", " "]
        speech_protect = INFO_NULL
    
    # Fallback
    elif not speech_protect: 
        if first_token in ["หอยอำพัน", "หอยเจดีย์ใหญ่", "หอยเจดีย์เล็ก", "หอยข้าวสารยอดมน", "ทากเล็บมือนาง"]:
            speech_protect = INFO_AMBER_SUBULI_MARTEN
        elif first_token == "หอยเลขหนึ่ง":
            speech_protect = INFO_JUMPING
        elif first_token in ["หอยดักดานหรือหอยทากสยาม", "หอยสาริกา"]:
            speech_protect = INFO_SARIKA
        elif first_token == "หอยทากยักษ์แอฟริกา":
            speech_protect = INFO_LISSA
        elif first_token in ["หอยเชอรี่", "หอยแอปเปิ้ลเกาะ"]:
            speech_protect = INFO_CHERRY
        elif first_token == "หอยสะดือบุ๋ม":
            speech_protect = INFO_BRADYBAENA
        elif first_token == "หอยโข่ง":
            speech_protect = INFO_PILA
        elif first_token in ["หอยคัน", "หอยคันหรือหอยคันอินโด", "หอยเจดีย์ลายเสือ"]:
            speech_protect = INFO_OTHER_WATER_SNAIL
        
        # เพิ่มเช็คตรงนี้เผื่อกรณีที่คำนำหน้าภาษาไทยตรงกับกรณี Null
        elif "ไม่ใช่หอยศัตรูพืช" in first_token:
            speech_protect = INFO_NULL

    return {
        "tokens": display_info,
        "protect_text": speech_protect,
        "thai_name": display_info[0],
        "sci_name_1": display_info[2] if len(display_info) > 2 else "",
        "sci_name_2": display_info[3] if len(display_info) > 3 else "",
        "family": display_info[5] if len(display_info) > 5 else ""
    }

def get_tts_html_script(snail_data, confidence_text, uploaded_file_exists):
    """
    สร้าง HTML Script สำหรับ TTS ผลลัพธ์ และปุ่มป้องกันกำจัด
    """
    
    # ประกาศตัวแปรค่าเริ่มต้นไว้ก่อนกัน Error
    result_text_th1 = ""
    result_text_thainame = ""
    result_text_en = ""
    result_text_introfam = ""
    result_text_family = ""
    result_text_th2 = ""
    result_text_introprotect = ""

    # ใช้ snail_data['tokens'][0] แทน display_info[0] และใช้ != แทน !==
    if snail_data['tokens'][0] != "ไม่ใช่หอยศัตรูพืช":
        result_text_th1 = " ผลการจัดจำแนกหอยศัตรูพืชจากรูป คาดว่าน่าจะเป็น"
        result_text_thainame = f" {snail_data['thai_name'] + (snail_data['tokens'][1] if len(snail_data['tokens'])>1 else '')}"
        result_text_en = f" {snail_data['sci_name_1'] + ' ' + snail_data['sci_name_2']}"
        result_text_introfam = f" {snail_data['tokens'][4] if len(snail_data['tokens'])>4 else ''}"
        result_text_family = f" family {snail_data['family']}"
        result_text_th2 = f" ด้วยมีความมั่นใจ {confidence_text}"
        result_text_introprotect = " ถ้าหาก คุณอยากทราบข้อมูลเกี่ยวกับวิธีการป้องกันกำจัด ผมสามารถให้ข้อมูลได้ โดยกดปุ่มป้องกัน กำจัดที่อยู่ด้านล่างสุดครับ"
    else:
        # กรณีที่เป็น Null (ไม่ใช่หอยศัตรูพืช) ให้พูดสั้นๆ
        result_text_th1 = " จากรูปภาพดังกล่าว ไม่ใช่หอยศัตรูพืชครับ"
        # ส่วนอื่นปล่อยว่างไว้

    html_code = f"""
    <script>
    function speakMultilingual() {{
            const synth = window.speechSynthesis;
                    
            const preloadUtterance = new SpeechSynthesisUtterance(" ");
            preloadUtterance.lang = "en-US";  
                    
            const part1 = new SpeechSynthesisUtterance({json.dumps(result_text_th1)});
            part1.lang = 'th-TH'; part1.pitch = 1.0; part1.rate = 0.75;
                    
            const part2 = new SpeechSynthesisUtterance({json.dumps(result_text_thainame)});
            part2.lang = 'th-TH'; part2.pitch = 1.0; part2.rate = 0.8;

            const part3 = new SpeechSynthesisUtterance({json.dumps(result_text_en)});
            part3.lang = 'en-US'; part3.pitch = 0.85; part3.rate = 0.7;
                    
            const part4 = new SpeechSynthesisUtterance({json.dumps(result_text_introfam)});
            part4.lang = 'th-TH'; part4.pitch = 1.0; part4.rate = 0.8;

            const part5 = new SpeechSynthesisUtterance({json.dumps(result_text_family)});
            part5.lang = 'en-US'; part5.pitch = 0.85; part5.rate = 0.7;
                    
            const part6 = new SpeechSynthesisUtterance({json.dumps(result_text_th2)});
            part6.lang = 'th-TH'; part6.pitch = 1.0; part6.rate = 0.8;
                    
            const part7 = new SpeechSynthesisUtterance({json.dumps(result_text_introprotect)});
            part7.lang = 'th-TH'; part7.pitch = 1.0; part7.rate = 0.8;
                
            function setVoicesAndSpeak() {{
                    const voices = synth.getVoices();
                    let thVoice = voices.find(v => v.lang === 'th-TH' && /female|หญิง|wom/i.test(v.name)) || voices.find(v => v.lang === 'th-TH');
                    let enVoice = voices.find(v => v.lang === 'en-US' && /female|หญิง|wom/i.test(v.name)) || voices.find(v => v.lang === 'en-US');

                    if (thVoice) {{
                        part1.voice = thVoice; part2.voice = thVoice;
                        part4.voice = thVoice; part6.voice = thVoice; part7.voice = thVoice;
                    }}
                    if (enVoice) {{
                        part3.voice = enVoice; part5.voice = enVoice; preloadUtterance.voice = enVoice;
                    }}

                    synth.cancel();
                    synth.speak(preloadUtterance);

                    preloadUtterance.onend = () => {{
                        setTimeout(() => {{
                            part1.onend = () => {{ setTimeout(() => synth.speak(part2), 200); }};
                            part2.onend = () => {{ setTimeout(() => synth.speak(part3), 1500); }};
                            part3.onend = () => {{ setTimeout(() => synth.speak(part4), 800); }};
                            part4.onend = () => {{ setTimeout(() => synth.speak(part5), 800); }};
                            part5.onend = () => {{ setTimeout(() => synth.speak(part6), 800); }};
                            part6.onend = () => {{ setTimeout(() => synth.speak(part7), 1500); }};
                            synth.speak(part1);
                        }}, 100);
                    }};
                }}

                if (synth.getVoices().length === 0) {{
                    synth.onvoiceschanged = setVoicesAndSpeak;
                }} else {{
                    setVoicesAndSpeak();
                }}
    }}

    function speechprotect() {{
            const synth = window.speechSynthesis;
            const msg1 = new SpeechSynthesisUtterance({json.dumps(snail_data['protect_text'])});
            msg1.lang = "th-TH";
            msg1.pitch = 1.0;
            msg1.rate = 0.8;

            function setVoiceAndSpeak() {{
                    const voices = synth.getVoices();
                    const thVoice = voices.find(v => v.lang === 'th-TH');
                    if (thVoice) {{
                        msg1.voice = thVoice;
                    }}
                    synth.cancel();
                    synth.speak(msg1);
                }}

                if (synth.getVoices().length === 0) {{
                    synth.onvoiceschanged = setVoiceAndSpeak;
                }} else {{
                    setVoiceAndSpeak();
                }}
    }}

    window.addEventListener('load', () => {{
        // Auto speak logic if needed
        {'speakMultilingual();' if uploaded_file_exists else ''}
    }});
    </script>
    <button onclick="speechprotect()" style="width: 220px; height: 65px; font-size: 20px;">🔊 การป้องกันและกำจัด</button>
    """

    return html_code





