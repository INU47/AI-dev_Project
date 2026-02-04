import logging
import torch
import numpy as np

logger = logging.getLogger("DecisionEngine")

class DecisionEngine:
    def __init__(self, cnn_model, lstm_model, atr_period=14):
        self.cnn = cnn_model
        self.lstm = lstm_model
        self.cnn.eval() # Set to inference mode
        self.lstm.eval()
        self.atr_period = atr_period

    def analyze(self, gaf_image, raw_series):
        """Standard single-TF analyze (Legacy/Compatibility)"""
        with torch.no_grad():
            cnn_logits = self.cnn(gaf_image)
            cnn_probs = torch.softmax(cnn_logits, dim=1)
            cnn_conf, cnn_class = torch.max(cnn_probs, 1)
            lstm_pred = self.lstm(raw_series)
            
            return {
                'class': cnn_class.item(),
                'conf': cnn_conf.item(),
                'trend': lstm_pred.item()
            }

    def analyze_mtf(self, mtf_data, ai_mode="CONSERVATIVE", exploration_rate=0.0):
        """
        mtf_data: { timeframe: (gaf_tensor, lstm_tensor) }
        Confluence Logic: 
        - CONSERVATIVE: Entry on M1 IF confirmed by M5 AND H1.
        - EXPLORER: Entry on M1 IF confirmed by M5 OR H1.
        """
        pattern_names = {
            0: "การพักฐาน / ไซด์เวย์ (Neutral/Consolidation)",
            1: "แรงซื้อสะสม / ขาขึ้น (Bullish Momentum)",
            2: "แรงขายกระจาย / ขาลง (Bearish Momentum)",
            3: "Hammer (Bullish Pin Bar) - สัญญาณกลับตัวจากแนวรับ",
            4: "Shooting Star (Bearish Pin Bar) - สัญญาณกลับตัวจากแนวต้าน",
            5: "Bullish Engulfing - การกลืนกินขาขึ้น (Strong Buy Reversal)",
            6: "Bearish Engulfing - การกลืนกินขาลง (Strong Sell Reversal)",
            7: "Morning Star - การกลับตัวจากระดับต่ำสุด",
            8: "Evening Star - การกลับตัวจากระดับสูงสุด",
            9: "Doji - ตลาดลังเลใจ (Indecision / Pivot Point)"
        }
        
        results = {}
        for tf, (gaf, seq) in mtf_data.items():
            results[tf] = self.analyze(gaf, seq)

        m1 = results.get(60)
        m5 = results.get(300)
        h1 = results.get(3600)

        signal = {'action': 'HOLD', 'confidence': 0.0, 'report': ''}
        
        # RL Exploration Logic (Priority 1: Experience gathering)
        if m1 and np.random.random() < exploration_rate:
            action = 'BUY' if np.random.random() > 0.5 else 'SELL'
            logger.info(f"🎲 RL Exploration Triggered: {action}")
            
            # Retrieve model inferences for logging (even if action is random)
            pattern_text = pattern_names.get(m1['class'], "Unknown")
            is_uptrend = m1['trend'] > 0
            future_outlook = "ขาขึ้นแข็งแกร่ง" if is_uptrend else "แนวโน้มขาลง"
            
            # Calculate Synthetic LSTM Confidence
            # Assumption: Trend is now scaled to Points (e.g. 10.0 = 1 Pip)
            # We treat > 50 points (5 pips) as "High Confidence" (1.0)
            trend_val = m1['trend']
            lstm_conf = min(abs(trend_val) / 50.0, 1.0)
            
            analyst_data = {
                'pattern': pattern_text,
                'future_outlook': future_outlook,
                'is_uptrend': is_uptrend,
                'confidence': m1['conf']
            }
            
            return {
                'action': action, 
                'confidence': 0.5, 
                'reason': 'RL Exploration',
                'report': f"🔍 **Exploration Mode**: ลองทดสอบคำสั่ง {action} เพื่อเรียนรู้ตลาด (Exploration Rate: {exploration_rate})",
                'analyst_metadata': analyst_data,
                'raw_cnn_class': m1['class'],  # Pass to DB
                'raw_lstm_trend': m1['trend'],
                'raw_lstm_conf': lstm_conf, # Now using calculated confidence
                'ai_mode': ai_mode
            }

        if m1:
            # 2. Confluence Logic
            is_bullish = m1['class'] == 1 and m1['trend'] > 0
            is_bearish = m1['class'] == 2 and m1['trend'] < 0
            
            # Additional Trend Confirmation (Optional in Explorer)
            if ai_mode == "EXPLORER":
                # If higher TFs exist, use them as filters, else allow M1 Solo
                bull_confirmed = True if not (m5 or h1) else ((m5['trend'] > 0 if m5 else False) or (h1['trend'] > 0 if h1 else False))
                bear_confirmed = True if not (m5 or h1) else ((m5['trend'] < 0 if m5 else False) or (h1['trend'] < 0 if h1 else False))
                
                # Confidence Threshold Relaxation
                min_conf = 0.6
            else:
                # Conservative remains strict: needs ALL
                if not (m5 and h1): return signal
                bull_confirmed = (m5['trend'] > 0 and h1['trend'] > 0)
                bear_confirmed = (m5['trend'] < 0 and h1['trend'] < 0)
                min_conf = 0.6

            # Generate Report Components
            pattern_text = pattern_names.get(m1['class'], "Unknown")
            is_uptrend = m1['trend'] > 0
            future_outlook = "ขาขึ้นแข็งแกร่ง" if is_uptrend else "แนวโน้มขาลง"
            confidence_pct = int(m1['conf'] * 100)

            # Metadata for Analyst
            analyst_data = {
                'pattern': pattern_text,
                'future_outlook': future_outlook,
                'is_uptrend': is_uptrend,
                'confidence': m1['conf']
            }
            
            # Helper for LSTM Confidence
            trend_val = m1['trend']
            lstm_conf = min(abs(trend_val) / 50.0, 1.0)

            if is_bullish and bull_confirmed and m1['conf'] >= min_conf:
                signal = {
                    'action': 'BUY', 
                    'confidence': m1['conf'], 
                    'reason': f'MTF Bullish {ai_mode}',
                    'analyst_metadata': analyst_data,
                    'raw_cnn_class': m1['class'],  
                    'raw_lstm_trend': m1['trend'],
                    'raw_lstm_conf': lstm_conf,
                    'ai_mode': ai_mode
                }
            elif is_bearish and bear_confirmed and m1['conf'] >= min_conf:
                signal = {
                    'action': 'SELL', 
                    'confidence': m1['conf'], 
                    'reason': f'MTF Bearish {ai_mode}',
                    'analyst_metadata': analyst_data,
                    'raw_cnn_class': m1['class'],  
                    'raw_lstm_trend': m1['trend'],
                    'raw_lstm_conf': lstm_conf,
                    'ai_mode': ai_mode
                }
            else:
                # Capture pattern detections even if no trade action is taken
                signal['analyst_metadata'] = analyst_data
                signal['raw_cnn_class'] = m1['class']
                signal['raw_lstm_trend'] = m1['trend']
                signal['raw_lstm_conf'] = lstm_conf
                
        return signal

    def extract_rl_features(self, gaf_image, raw_series):
        """
        Exposes latent features for Reinforcement Learning.
        Returns a flat numpy array (State Vector).
        """
        with torch.no_grad():
            # Get CNN Features (before final linear layer)
            # PatternCNN.forward has x = self.fc2(x), let's assume we want fc1 or similar
            # Since models.py is simple, we'll just take the logits + trend as a basic state for now
            cnn_logits = self.cnn(gaf_image)
            lstm_pred = self.lstm(raw_series)
            
            # Combine into a vector
            state = torch.cat([cnn_logits.flatten(), lstm_pred.flatten()])
            return state.cpu().numpy()

class RiskManager:
    def __init__(self, risk_per_trade=0.01):
        self.risk_per_trade = risk_per_trade 

    def calculate_sl_tp(self, symbol, action, entry_price, stop_loss_pips=20, reward_ratio=2.0, point=0.00001, tick_size=0.00001):
        """Calculates precise price levels for SL and TP based on pips and tick_size"""
        # Determine pip size (Standard pip = 10 * point for 5-digit brokers)
        # For JPY/XAU, point is usually 0.01 or 0.001. 
        # A 'pip' is generally 10 points on 5-digit brokers.
        pip_value = 10 * point
        
        sl_offset = stop_loss_pips * pip_value
        tp_offset = sl_offset * reward_ratio
        
        if action.upper() == "BUY":
            sl = entry_price - sl_offset
            tp = entry_price + tp_offset
        else:
            sl = entry_price + sl_offset
            tp = entry_price - tp_offset
            
        # Standardize rounding to the broker's tick_size
        def normalize(price):
            return round(price / tick_size) * tick_size
            
        return normalize(sl), normalize(tp)

    def calculate_lot_size(self, account_equity, stop_loss_pips, confidence=0.5, pip_value=10):
        if stop_loss_pips <= 0: return 0.01
        
        # Dynamic Sizing: Scale risk based on confidence
        # Base Risk: 1%
        # If Confidence > 0.8: Risk -> 1.5%
        # If Confidence < 0.6: Risk -> 0.5%
        
        confidence_factor = max(0.5, min(1.5, confidence / 0.7)) # Normalize around 0.7
        adjusted_risk = self.risk_per_trade * confidence_factor
        
        risk_amount = account_equity * adjusted_risk
        raw_lots = risk_amount / (stop_loss_pips * pip_value)
        return round(raw_lots, 2)
