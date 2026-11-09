from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from pypdf import PdfReader
import io
import json
from google import genai

# 1. โหลดค่าจาก .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Warning: ไม่พบ GEMINI_API_KEY ในไฟล์ .env")

client = genai.Client(api_key=api_key)

# 2. ประกาศสร้าง FastAPI app
app = FastAPI(title="UniTrack AI Backend (Gemini Edition)", version="1.0")

# 3. ตั้งค่า CORS ให้ Frontend เรียกใช้งานได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "UniTrack AI Backend (Gemini) is running successfully!"}

@app.post("/api/analyze")
async def analyze_document(
    purposes: str = Form(...),
    keywords: str = Form(...),
    file: UploadFile = File(None),
    urls: str = Form(None)
):
    try:
        parsed_purposes = json.loads(purposes)
        parsed_keywords = json.loads(keywords)
        parsed_urls = json.loads(urls) if urls and urls != "null" else []
    except Exception:
        raise HTTPException(status_code=400, detail="รูปแบบข้อมูลไม่ถูกต้อง")

    pdf_text_by_page = {}
    combined_content = ""

    # 1. ถ้ามีการอัปโหลดไฟล์ PDF มาด้วย
    if file:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ PDF เท่านั้น")
        
        contents = await file.read()
        try:
            reader = PdfReader(io.BytesIO(contents))
            for index, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pdf_text_by_page[index + 1] = text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ไม่สามารถอ่านไฟล์ PDF ได้: {str(e)}")

        for page_num, text in pdf_text_by_page.items():
            combined_content += f"\n--- หน้า {page_num} ---\n{text}"

    # 2. ถ้ามีลิงก์เว็บไซต์ที่ผู้ใช้กรอกเข้ามา
    if parsed_urls:
        combined_content += f"\n\nแหล่งข้อมูลจากลิงก์เว็บไซต์ที่ระบุ:\n" + "\n".join([f"- {u}" for u in parsed_urls])

    # ตรวจสอบว่ามีเนื้อหาหรือลิงก์หรือไม่
    if not combined_content.strip():
        raise HTTPException(status_code=400, detail="ไม่พบข้อมูลภายในไฟล์ PDF หรือลิงก์เว็บไซต์ที่ระบุ")

    # สร้าง Prompt สำหรับสั่งงาน Gemini API
    prompt = f"""
คุณคือผู้ช่วยอัจฉริยะ UniTrack AI ทำหน้าที่วิเคราะห์เอกสารรับสมัครเข้ามหาวิทยาลัย
จุดประสงค์การค้นหา: {', '.join(parsed_purposes)}
คำค้นหาที่ต้องการเน้น: {', '.join(parsed_keywords)}

เนื้อหาจากเอกสาร/ลิงก์:
{combined_content}

คำสั่ง:
1. ค้นหาเฉพาะข้อความ ข้อมูล หรือตัวเลขในเอกสารที่ตรงกับ "คำค้นหา" และ "จุดประสงค์" ข้างต้นเท่านั้น
2. ห้ามแต่งเติมข้อมูลหรือสร้างแหล่งอ้างอิงขึ้นเอง
3. รูปแบบการตอบ: สรุปให้กระชับ ตรงประเด็น และ **ต้องระบุเลขหน้า (เช่น หน้า 3) หรือข้อความอ้างอิงต้นฉบับ** กำกับไว้ทุกจุดที่พบข้อมูล
4. หากคำค้นหาใดไม่พบในเอกสาร ให้แจ้งว่าไม่พบ ไม่ต้องแต่งข้อมูลขึ้นมา
"""

    try:
        # อัปเดตใช้โมเดล gemini-3.8-flash ตามที่ระบบแนะนำ
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        
        ai_result = response.text

        return {
            "status": "success",
            "source_type": "pdf" if file else "url",
            "analysis_result": ai_result,
            "pages_read": list(pdf_text_by_page.keys())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini: {str(e)}")