import google.generativeai as genai
import logging
import json

logger = logging.getLogger("VirtualAnalyst")

class VirtualAnalyst:
    def __init__(self, api_key, model_name="gemini-pro"):
        if not api_key:
            logger.error("Gemini API Key missing!")
            self.model = None
            return
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Virtual Analyst Initialized with model: {model_name}")

    async def generate_report(self, signal_data, mode="telegram"):
        """
        Generates a technical analysis report.
        mode: "telegram" (concise) or "dashboard" (trader log style)
        """
        if not self.model:
            return "⚠️ [Analyst Offline] ไม่สามารถสร้างบทวิเคราะห์ได้ในขณะนี้"

        if mode == "dashboard":
            prompt = f"""
            คุณคือ "Senior Quant Trader" ที่มีประสบการณ์สูง
            ภารกิจ: เขียน Log บันทึกการเทรดสั้นๆ ลงใน Dashboard ส่วนตัว เพื่อเตือนความจำ
            
            ข้อมูล:
            - คู่เงิน: {signal_data.get('symbol')}
            - Action: {signal_data.get('action')}
            - Pattern: {signal_data.get('pattern')}
            - Confidence: {int(signal_data.get('confidence', 0) * 100)}%
            - Outlook: {signal_data.get('future_outlook')}
            - Price: {signal_data.get('price')}

            สไตล์การเขียน:
            - เขียนเหมือนคุยกับตัวเอง หรือจดบันทึกความคิด (Think Aloud)
            - ใช้ศัพท์ Trader (Support, Resistance, Rejection, Volume)
            - วิเคราะห์เหตุผลว่าทำไมถึงเข้าออเดอร์นี้ สั้นๆ 2-3 ประโยค
            - ไม่ต้องขึ้นต้นว่า "สรุป" หรือ "สวัสดี" ใส่เนื้อหาเลย
            
            ตัวอย่าง:
            "เจอ Bullish Engulfing ที่โซนรับสำคัญ M1 ราคามีการ Rejection สวยๆ วอลุ่มเข้า Support เอานะ ไม้นี้มั่นใจ 85% ลุย!"
            """
        else:
            # Telegram Mode (Original)
            prompt = f"""
            คุณคือ "Professional Quant Analyst"
            ภารกิจ: สรุปบทวิเคราะห์จาก AI ให้กระชับที่สุด (Glanceable) สำหรับ Telegram
    
            ข้อมูล:
            - คู่เงิน: {signal_data.get('symbol')}
            - คำแนะนำ: {signal_data.get('action')}
            - รูปแบบ: {signal_data.get('pattern')}
            - ความมั่นใจ: {int(signal_data.get('confidence', 0) * 100)}%
            - แนวโน้ม: {signal_data.get('future_outlook')}
    
            กรุณาเขียนสรุป 3 บรรทัดดังนี้ (ห้ามใช้ตัวอักษรพิเศษที่อาจทำลาย Markdown):
            🎯 Signal: [Action] [Symbol] ([Confidence]%)
            📊 Reason: [วิเคราะห์สั้นๆ 1 ประโยค]
            ⚠️ Risk: [คำแนะนำความเสี่ยงสั้นๆ]
    
            *ใช้ภาษาเป็นกันเองแต่ดูเป็นมืออาชีพ ไม่เวิ่นเว้อ*
            """

        try:
            # Note: The google-generativeai library's async support might vary, 
            # using sync call in a thread or direct if supported. 
            # For simplicity in this environment, we'll use generate_content.
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                logger.warning("Gemini API Quota Exceeded. Analyst is silent.")
                return "⏸️ [Analyst Sleep] เกินขีดจำกัดการใช้งานฟรี (Quota Exceeded) ระบบจะกลับมาทำงานอัตโนมัติเมื่อครบกำหนดเวลาครับ"
            
            logger.error(f"Error generating LLM report: {e}")
            return f"⚠️ [Analyst Error] ไม่สามารถสร้างบทวิเคราะห์ได้ (Error: {err_msg[:50]}...)"
