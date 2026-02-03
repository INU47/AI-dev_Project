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

    async def generate_report(self, signal_data):
        """
        Generates a professional Thai technical analysis report.
        signal_data: {symbol, action, confidence, pattern, future_outlook, price, volume}
        """
        if not self.model:
            return "⚠️ [Analyst Offline] ไม่สามารถสร้างบทวิเคราะห์ได้ในขณะนี้"

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
