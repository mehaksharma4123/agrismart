import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, io, requests
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except:
    SR_AVAILABLE = False

# -----------------------
# Load model & encoder (fixed path)
# -----------------------
@st.cache_data(show_spinner=False)
def load_model_and_encoder(model_path="models/crop_model_pipeline.pkl", le_path="models/label_encoder.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        return None, None
    m = joblib.load(model_path)
    le = joblib.load(le_path)
    return m, le

model, le = load_model_and_encoder()

st.set_page_config(page_title="AgriSmart", page_icon="🌱", layout="wide")
def get_model_classes(m):
    if m is None:
        return None
    try:
        if hasattr(m, "named_steps") and "clf" in m.named_steps:
            clf = m.named_steps["clf"]
            if hasattr(clf, "classes_"):
                return list(clf.classes_)
    except Exception:
        pass
    # Direct estimator
    if hasattr(m, "classes_"):
        return list(m.classes_)
    return None

def predict_with_model(m, sample_df):
    """Return (pred_label, proba_array, classes) or (None, None, None) on failure"""
    if m is None:
        return None, None, None
    try:
        pred = m.predict(sample_df)[0]
    except Exception:
        return None, None, None
    proba = None
    try:
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(sample_df)
        elif hasattr(m, "named_steps") and "clf" in m.named_steps and hasattr(m.named_steps["clf"], "predict_proba"):
            proba = m.named_steps["clf"].predict_proba(m.named_steps.get("preprocessor").transform(sample_df) if "preprocessor" in m.named_steps else sample_df)
    except Exception:
        proba = None

    classes = get_model_classes(m)
    return pred, proba, classes

def fetch_weather_openweathermap(city):
    key = "b21c29109fb8ac7f3f3d9bfc7ba06935" 
    try:
        r = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric", timeout=6)
        j = r.json()
        if j.get("main"):
            return j
    except Exception:
        return None
    return None
TRANSLATIONS = {
    "English": {
        "title": "🌾 AgriSmart: AI-powered Crop Advisory",
        "subtitle": "Crop selection, soil health, NPK guidance, pest detection, weather & voice support.",
        "predict": "🔍 Predict Best Crop",
        "soil_health": "🧪 Soil Health & Fertilizer Guidance",
        "weather": "⛈ Weather Insights",
        "market": "💹 Selling Prices",
        "pest": "🐛 Pest & Disease Detection",
        "feedback": "💬 Farmer Feedback",
        "voice": "🎙 Voice Support",
        "profit": "💰 Profit Analysis",
        "recommended": "🌟 Recommended Crops",
        "best": "Best Recommendation",
        "other": "Other Good Options",
        "inputs": "📥 Enter Your Farm Details",
        "soil_moisture": "Soil Moisture (%)",
        "soil_ph": "Soil pH",
        "soil_type": "Soil Type",
        "temp": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "land_area": "Land Area (acres)",
        "yield": "Expected Yield (qtl/acre)",
        "nitrogen": "Nitrogen",
        "phosphorus": "Phosphorus",
        "potassium": "Potassium",
        "magnesium": "Magnesium",
        "zinc": "Zinc",
        "hardness": "Soil Hardness (%)",
        "profit_form": "Custom Profit Analysis",
        "select_crop": "Select Crop for Profit Analysis",
        "cost_per_quintal": "Input Cost per Quintal (₹)",
        "market_price_input": "Selling Price per Quintal (₹)",
        "calc_profit": "Calculate Profit/Loss",
        "acidic": "⚠ Acidic soil",
        "alkaline": "⚠ Alkaline soil",
        "good_ph": "✅ pH is good",
        "upload_leaf": "Upload leaf",
        "pest_msg": "⚠ Pest AI model coming soon.",
        "feedback_input": "Enter feedback",
        "feedback_btn": "Submit Feedback",
        "feedback_success": "✅ Thank you for your feedback!",
        "prediction_failed": "Prediction failed",
        "selling_price": "Selling Price",
        "input_cost": "Input Cost",
        "total_input_cost": "Total Input Cost",
        "revenue": "Total Revenue",
        "profit_result": "Net Profit",
        "loss_result": "Net Loss",
        "price_col": "Selling Price (₹/qtl)",
        "crop_col": "Crop",
        "city": "📍 City",
        "adv_toggle": "Enable Mg/Zn/Hardness",
        "voice_out_toggle": "Enable Voice Output",
        "voice_in_toggle": "Enable Voice Input (STT)",
        "voice_failed": "Voice failed",
        "top_crops_voice": "Top crops",
        "crop_names": {
            "rice": "Rice", "wheat": "Wheat", "maize": "Maize",
            "sugarcane": "Sugarcane", "cotton": "Cotton",
            "potato": "Potato", "Soybean": "Soybean","adzuki beans":"Adzuki Beans","apple":"Apple",
            "banana":"Banana","black gram":"Black Gram","chickpea":"Chickpea","coconut":"Coconut",
            "coffee":"Coffee","grapes":"Grapes","ground nut":"Ground Nut","jute":"Jute","kidney beans":
            "Kidney Beans","lentil":"Lentil","mango":"Mango","millet":"Millet","moth beans":"Moth Beans",
            "mung bean":"Mung Beans","muskmelon":"Muskmelon","orange":"Orange","papaya":"Papaya",
            "peas":"Peas","pigeon peas":"Pigeon Peas","pomegranate":"Pomegranate","rubber":"rubber",
            "tea":"Tea","tobacco":"Tobacco","watermelon":"Watermelon"
        },
        "desc": {
            "rice": "Rice — staple crop",
            "wheat": "Wheat — cereal crop",
            "maize": "Maize — versatile",
            "sugarcane": "Sugarcane — cash crop",
            "cotton": "Cotton — fiber crop",
            "potato": "Potato — root crop",
            "Soybean": "Soybean — oilseed crop"
        }
    },
    "हिन्दी": {
        "title": "🌾 एग्रीस्मार्ट: एआई आधारित फसल सलाह",
        "subtitle": "फसल चयन, मिट्टी स्वास्थ्य, NPK सलाह, कीट पहचान, मौसम और वॉइस सपोर्ट।",
        "predict": "🔍 सबसे अच्छी फसल बताएं",
        "soil_health": "🧪 मिट्टी का स्वास्थ्य और उर्वरक सलाह",
        "weather": "⛈ मौसम जानकारी",
        "market": "💹 बिक्री मूल्य",
        "pest": "🐛 कीट और रोग पहचान",
        "feedback": "💬 किसान प्रतिक्रिया",
        "voice": "🎙 वॉइस सपोर्ट",
        "profit": "💰 लाभ विश्लेषण",
        "recommended": "🌟 सुझाई गई फसलें",
        "best": "सर्वोत्तम सिफारिश",
        "other": "अन्य अच्छे विकल्प",
        "inputs": "📥 अपनी खेती की जानकारी दर्ज करें",
        "soil_moisture": "मिट्टी में नमी (%)",
        "soil_ph": "मिट्टी का pH",
        "soil_type": "मिट्टी का प्रकार",
        "temp": "तापमान (°C)",
        "humidity": "नमी (%)",
        "land_area": "भूमि क्षेत्र (एकड़)",
        "yield": "अनुमानित उत्पादन (क्विंटल/एकड़)",
        "nitrogen": "नाइट्रोजन",
        "phosphorus": "फॉस्फोरस",
        "potassium": "पोटाशियम",
        "magnesium": "मैग्नीशियम",
        "zinc": "जिंक",
        "hardness": "मिट्टी की कठोरता (%)",
        "profit_form": "कस्टम लाभ विश्लेषण",
        "select_crop": "लाभ विश्लेषण के लिए फसल चुनें",
        "cost_per_quintal": "प्रति क्विंटल इनपुट लागत (₹)",
        "market_price_input": "प्रति क्विंटल बिक्री मूल्य (₹)",
        "calc_profit": "लाभ/हानि निकालें",
        "acidic": "⚠ अम्लीय मिट्टी",
        "alkaline": "⚠ क्षारीय मिट्टी",
        "good_ph": "✅ pH अच्छा है",
        "upload_leaf": "पत्ता अपलोड करें",
        "pest_msg": "⚠ कीट एआई मॉडल जल्द ही आ रहा है।",
        "feedback_input": "प्रतिक्रिया दर्ज करें",
        "feedback_btn": "प्रतिक्रिया भेजें",
        "feedback_success": "✅ आपकी प्रतिक्रिया के लिए धन्यवाद!",
        "prediction_failed": "भविष्यवाणी असफल",
        "selling_price": "बिक्री मूल्य",
        "input_cost": "इनपुट लागत",
        "total_input_cost": "कुल इनपुट लागत",
        "revenue": "कुल राजस्व",
        "profit_result": "शुद्ध लाभ",
        "loss_result": "शुद्ध हानि",
        "price_col": "बिक्री मूल्य (₹/क्विंटल)",
        "crop_col": "फसल",
        "city": "📍 शहर",
        "adv_toggle": "Mg/Zn/कठोरता सक्षम करें",
        "voice_out_toggle": "वॉइस आउटपुट सक्षम करें",
        "voice_in_toggle": "वॉइस इनपुट (STT) सक्षम करें",
        "voice_failed": "वॉइस विफल",
        "top_crops_voice": "सर्वोच्च फसलें",
        "crop_names": {
    "rice": "धान", "wheat": "गेहूं", "maize": "मक्का",
    "sugarcane": "गन्ना", "cotton": "कपास",
    "potato": "आलू", "Soybean": "सोयाबीन", "adzuki beans": "अड़द की फलियाँ", 
    "apple": "सेब", "banana": "केला", "black gram": "उड़द", 
    "chickpea": "चना", "coconut": "नारियल", "coffee": "कॉफी",
    "grapes": "अंगूर", "ground nut": "मूंगफली", "jute": "जूट", 
    "kidney beans": "राजमा", "lentil": "मसूर", "mango": "आम", 
    "millet": "बाजरा", "moth beans": "मटकी", "mung bean": "मूंग", 
    "muskmelon": "खरबूजा", "orange": "संतरा", "papaya": "पपीता", 
    "peas": "मटर", "pigeon peas": "अरहर", "pomegranate": "अनार", 
    "rubber": "रबर", "tea": "चाय", "tobacco": "तंबाकू", 
    "watermelon": "तरबूज"
        },
        "desc": {
            "rice": "धान — मुख्य फसल",
            "wheat": "गेहूं — अनाज फसल",
            "maize": "मक्का — बहुउपयोगी",
            "sugarcane": "गन्ना — नगदी फसल",
            "cotton": "कपास — रेशेदार फसल",
            "potato": "आलू — कंद फसल",
            "Soybean": "सोयाबीन — तिलहन फसल"
        }
    },
    "ਪੰਜਾਬੀ": {
        "title": "🌾 ਏਗ੍ਰੀਸਮਾਰਟ: ਏਆਈ ਆਧਾਰਿਤ ਫਸਲ ਸਲਾਹ",
        "subtitle": "ਫਸਲ ਚੋਣ, ਮਿੱਟੀ ਸਿਹਤ, NPK ਸਲਾਹ, ਕੀੜੇ/ਬਿਮਾਰੀ ਪਛਾਣ, ਮੌਸਮ ਅਤੇ ਆਵਾਜ਼ ਸਹਾਇਤਾ।",
        "predict": "🔍 ਸਭ ਤੋਂ ਵਧੀਆ ਫਸਲ ਦਿਖਾਓ",
        "soil_health": "🧪 ਮਿੱਟੀ ਦੀ ਸਿਹਤ ਅਤੇ ਖਾਦ ਸਲਾਹ",
        "weather": "⛈ ਮੌਸਮ ਜਾਣਕਾਰੀ",
        "market": "💹 ਵਿਕਰੀ ਕੀਮਤ",
        "pest": "🐛 ਕੀੜੇ ਅਤੇ ਬਿਮਾਰੀ ਦੀ ਪਛਾਣ",
        "feedback": "💬 ਕਿਸਾਨ ਫੀਡਬੈਕ",
        "voice": "🎙 ਆਵਾਜ਼ ਸਹਾਇਤਾ",
        "profit": "💰 ਨਫ਼ਾ ਵਿਸ਼ਲੇਸ਼ਣ",
        "recommended": "🌟 ਸੁਝਾਈਆਂ ਫਸਲਾਂ",
        "best": "ਸਭ ਤੋਂ ਵਧੀਆ ਸਿਫਾਰਸ਼",
        "other": "ਹੋਰ ਚੰਗੇ ਵਿਕਲਪ",
        "inputs": "📥 ਆਪਣੀ ਖੇਤੀ ਦੀ ਜਾਣਕਾਰੀ ਦਿਓ",
        "soil_moisture": "ਮਿੱਟੀ ਵਿਚ ਨਮੀ (%)",
        "soil_ph": "ਮਿੱਟੀ ਦਾ pH",
        "soil_type": "ਮਿੱਟੀ ਦਾ ਕਿਸਮ",
        "temp": "ਤਾਪਮਾਨ (°C)",
        "humidity": "ਨਮੀ (%)",
        "land_area": "ਜ਼ਮੀਨ ਖੇਤਰ (ਏਕੜ)",
        "yield": "ਅਨੁਮਾਨਿਤ ਪੈਦਾਵਾਰ (ਕੁਇੰਟਲ/ਏਕੜ)",
        "nitrogen": "ਨਾਈਟਰੋਜਨ",
        "phosphorus": "ਫਾਸਫੋਰਸ",
        "potassium": "ਪੋਟਾਸੀਅਮ",
        "magnesium": "ਮੈਗਨੀਸ਼ੀਅਮ",
        "zinc": "ਜ਼ਿੰਕ",
        "hardness": "ਮਿੱਟੀ ਦੀ ਸਖ਼ਤੀ (%)",
        "profit_form": "ਕਸਟਮ ਨਫ਼ਾ ਵਿਸ਼ਲੇਸ਼ਣ",
        "select_crop": "ਨਫ਼ਾ ਵਿਸ਼ਲੇਸ਼ਣ ਲਈ ਫਸਲ ਚੁਣੋ",
        "cost_per_quintal": "ਪ੍ਰਤੀ ਕੁਇੰਟਲ ਇਨਪੁੱਟ ਲਾਗਤ (₹)",
        "market_price_input": "ਪ੍ਰਤੀ ਕੁਇੰਟਲ ਵਿਕਰੀ ਕੀਮਤ (₹)",
        "calc_profit": "ਨਫ਼ਾ/ਨੁਕਸਾਨ ਕੱਡੋ",
        "acidic": "⚠ ਅਮਲੀ ਮਿੱਟੀ",
        "alkaline": "⚠ ਖਾਰਾ ਮਿੱਟੀ",
        "good_ph": "✅ pH ਵਧੀਆ ਹੈ",
        "upload_leaf": "ਪੱਤਾ ਅਪਲੋਡ ਕਰੋ",
        "pest_msg": "⚠ ਕੀੜਾ AI ਮਾਡਲ ਜਲਦੀ ਆ ਰਿਹਾ ਹੈ।",
        "feedback_input": "ਫੀਡਬੈਕ ਦਿਓ",
        "feedback_btn": "ਫੀਡਬੈਕ ਭੇਜੋ",
        "feedback_success": "✅ ਤੁਹਾਡੇ ਫੀਡਬੈਕ ਲਈ ਧੰਨਵਾਦ!",
        "prediction_failed": "ਅਨੁਮਾਨ ਫੇਲ੍ਹ",
        "selling_price": "ਵਿਕਰੀ ਕੀਮਤ",
        "input_cost": "ਇਨਪੁੱਟ ਲਾਗਤ",
        "total_input_cost": "ਕੁੱਲ ਇਨਪੁੱਟ ਲਾਗਤ",
        "revenue": "ਕੁੱਲ ਆਮਦਨ",
        "profit_result": "ਖਾਲਿਸ ਨਫ਼ਾ",
        "loss_result": "ਖਾਲਿਸ ਨੁਕਸਾਨ",
        "price_col": "ਵਿਕਰੀ ਕੀਮਤ (₹/ਕੁਇੰਟਲ)",
        "crop_col": "ਫਸਲ",
        "city": "📍 ਸ਼ਹਿਰ",
        "adv_toggle": "Mg/Zn/ਸਖ਼ਤੀ ਸਮਰਥਨ ਕਰੋ",
        "voice_out_toggle": "ਆਵਾਜ਼ ਆਉਟਪੁੱਟ ਸਮਰਥਨ ਕਰੋ",
        "voice_in_toggle": "ਆਵਾਜ਼ ਇਨਪੁੱਟ (STT) ਸਮਰਥਨ ਕਰੋ",
        "voice_failed": "ਆਵਾਜ਼ ਫੇਲ੍ਹ ਹੋਈ",
        "top_crops_voice": "ਸਿਖਰ ਫਸਲਾਂ",
        "crop_names":{
    "rice": "ਚੌਲ", "wheat": "ਗੇਹੂੰ", "maize": "ਮੱਕੀ",
    "sugarcane": "ਗੰਨਾ", "cotton": "ਕਪਾਹ",
    "potato": "ਆਲੂ", "Soybean": "ਸੋਯਾਬੀਨ", "adzuki beans": "ਅੜਦ ਦੀਆਂ ਫਲੀਆਂ", 
    "apple": "ਸੇਬ", "banana": "ਕੇਲਾ", "black gram": "ਉੜਦ", 
    "chickpea": "ਚਨਾ", "coconut": "ਨਾਰੀਅਲ", "coffee": "ਕੌਫੀ",
    "grapes": "ਅੰਗੂਰ", "ground nut": "ਮੂੰਗਫਲੀ", "jute": "ਜੂਟ", 
    "kidney beans": "ਰਾਜਮਾ", "lentil": "ਮਸੂਰ", "mango": "ਆਮ", 
    "millet": "ਬਾਜਰਾ", "moth beans": "ਮਟਕੀ", "mung bean": "ਮੂੰਗ", 
    "muskmelon": "ਖਰਬੂਜ਼ਾ", "orange": "ਸੰਤਰਾ", "papaya": "ਪਪੀਤਾ", 
    "peas": "ਮਟਰ", "pigeon peas": "ਅਰਹਰ", "pomegranate": "ਅਨਾਰ", 
    "rubber": "ਰਬੜ", "tea": "ਚਾਹ", "tobacco": "ਤੰਬਾਕੂ", 
    "watermelon": "ਤਰਬੂਜ਼"
},
        "desc": {
            "rice": "ਚਾਵਲ — ਮੁੱਖ ਫਸਲ",
            "wheat": "ਗੰਧਮ — ਅਨਾਜ",
            "maize": "ਮੱਕੀ — ਬਹੁ-ਉਪਯੋਗੀ",
            "sugarcane": "ਗੰਨਾ — ਨਗਦੀ ਫਸਲ",
            "cotton": "ਕਪਾਹ — ਰੇਸ਼ੇਦਾਰ ਫਸਲ",
            "potato": "ਆਲੂ — ਜੜ ਫਸਲ",
            "Soybean": "ਸੋਇਆਬੀਨ — ਤੇਲਹਨ ਫਸਲ"
        }
    },
    "বাংলা": {
        "title": "🌾 AgriSmart: এআই-চালিত ফসল পরামর্শ",
        "subtitle": "ফসল নির্বাচন, মাটির স্বাস্থ্য, NPK নির্দেশনা, কীটপতঙ্গ সনাক্তকরণ, আবহাওয়া ও ভয়েস সহায়তা।",
        "predict": "🔍 সেরা ফসল পূর্বানুমান",
        "soil_health": "🧪 মাটির স্বাস্থ্য ও সার নির্দেশনা",
        "weather": "⛈ আবহাওয়া অন্তর্দৃষ্টি",
        "market": "💹 বিক্রয়মূল্য",
        "pest": "🐛 কীটপতঙ্গ ও রোগ সনাক্তকরণ",
        "feedback": "💬 কৃষকের মতামত",
        "voice": "🎙 ভয়েস সহায়তা",
        "profit": "💰 লাভ বিশ্লেষণ",
        "recommended": "🌟 প্রস্তাবিত ফসল",
        "best": "সেরা পরামর্শ",
        "other": "অন্য ভালো বিকল্প",
        "inputs": "📥 আপনার খামারের তথ্য দিন",
        "soil_moisture": "মাটির আর্দ্রতা (%)",
        "soil_ph": "মাটির pH",
        "soil_type": "মাটির ধরন",
        "temp": "তাপমাত্রা (°C)",
        "humidity": "আর্দ্রতা (%)",
        "land_area": "জমির আয়তন (একর)",
        "yield": "প্রত্যাশিত উৎপাদন (কুইন্টাল/একর)",
        "nitrogen": "নাইট্রোজেন",
        "phosphorus": "ফসফরাস",
        "potassium": "পটাশিয়াম",
        "magnesium": "ম্যাগনেসিয়াম",
        "zinc": "জিঙ্ক",
        "hardness": "মাটির কঠোরতা (%)",
        "profit_form": "কাস্টম লাভ বিশ্লেষণ",
        "select_crop": "লাভ বিশ্লেষণের জন্য ফসল নির্বাচন করুন",
        "cost_per_quintal": "প্রতি কুইন্টালের ইনপুট খরচ (₹)",
        "market_price_input": "প্রতি কুইন্টালের বিক্রয়মূল্য (₹)",
        "calc_profit": "লাভ/ক্ষতি হিসাব করুন",
        "acidic": "⚠ অম্লীয় মাটি",
        "alkaline": "⚠ ক্ষারীয় মাটি",
        "good_ph": "✅ pH ভালো আছে",
        "upload_leaf": "পাতা আপলোড করুন",
        "pest_msg": "⚠ কীটপতঙ্গ এআই মডেল শীঘ্রই আসছে।",
        "feedback_input": "মতামত লিখুন",
        "feedback_btn": "মতামত জমা দিন",
        "feedback_success": "✅ আপনার মতামতের জন্য ধন্যবাদ!",
        "prediction_failed": "পূর্বানুমান ব্যর্থ হয়েছে",
        "selling_price": "বিক্রয়মূল্য",
        "input_cost": "ইনপুট খরচ",
        "total_input_cost": "মোট ইনপুট খরচ",
        "revenue": "মোট আয়",
        "profit_result": "নেট লাভ",
        "loss_result": "নেট ক্ষতি",
        "price_col": "বিক্রয়মূল্য (₹/কুইন্টাল)",
        "crop_col": "ফসল",
        "city": "📍 শহর",
        "adv_toggle": "Mg/Zn/কঠোরতা সক্রিয় করুন",
        "voice_out_toggle": "ভয়েস আউটপুট সক্রিয় করুন",
        "voice_in_toggle": "ভয়েস ইনপুট (STT) সক্রিয় করুন",
        "voice_failed": "ভয়েস ব্যর্থ হয়েছে",
        "top_crops_voice": "শীর্ষ ফসল",
        "crop_names": {
    "rice": "ধান", "wheat": "গম", "maize": "ভুট্টা",
    "sugarcane": "আখ", "cotton": "সুতির", 
    "potato": "আলু", "Soybean": "সয়াবিন", "adzuki beans": "অড়হর শিম", 
    "apple": "আপেল", "banana": "কলা", "black gram": "কালো মসুর", 
    "chickpea": "ছোলা", "coconut": "নারকেল", "coffee": "কফি", 
    "grapes": "আঙুর", "ground nut": "চীনাবাদাম", "jute": "পাট", 
    "kidney beans": "রাজমা", "lentil": "মসুর ডাল", "mango": "আম", 
    "millet": "বাজরা", "moth beans": "মথ ডাল", "mung bean": "মুগ ডাল", 
    "muskmelon": "খরবুজা", "orange": "কমলা", "papaya": "পেঁপে", 
    "peas": "মটরশুঁটি", "pigeon peas": "তুর ডাল", "pomegranate": "ডালিম", 
    "rubber": "রাবার", "tea": "চা", "tobacco": "তামাক", 
    "watermelon": "তরমুজ"
},
        "desc": {
            "rice": "ধান — প্রধান খাদ্যশস্য",
            "wheat": "গম — শস্য ফসল",
            "maize": "ভুট্টা — বহুমুখী ফসল",
            "sugarcane": "আখ — নগদ ফসল",
            "cotton": "সুতির ফসল — তন্তু ফসল",
            "potato": "আলু — কন্দজাত ফসল",
            "Soybean": "সয়াবিন — তৈলবীজ ফসল"
        }
    },
    "ગુજરાતી": {
        "title": "🌾 AgriSmart: એઆઈ આધારિત પાક સલાહ",
        "subtitle": "પાક પસંદગી, માટીની તંદુરસ્તી, NPK માર્ગદર્શન, જીવાત શોધ, હવામાન અને વોઇસ સપોર્ટ.",
        "predict": "🔍 શ્રેષ્ઠ પાકની આગાહી કરો",
        "soil_health": "🧪 માટીની તંદુરસ્તી અને ખાતર માર્ગદર્શન",
        "weather": "⛈ હવામાનની માહિતી",
        "market": "💹 વેચાણના ભાવ",
        "pest": "🐛 જીવાત અને રોગ શોધ",
        "feedback": "💬 ખેડૂત પ્રતિસાદ",
        "voice": "🎙 વોઇસ સપોર્ટ",
        "profit": "💰 નફાનો વિશ્લેષણ",
        "recommended": "🌟 ભલામણ કરેલા પાક",
        "best": "શ્રેષ્ઠ ભલામણ",
        "other": "અન્ય સારા વિકલ્પો",
        "inputs": "📥 તમારી ખેતીની માહિતી દાખલ કરો",
        "soil_moisture": "માટીની ભેજ (%)",
        "soil_ph": "માટીની pH",
        "soil_type": "માટીની જાત",
        "temp": "તાપમાન (°C)",
        "humidity": "આર્દ્રતા (%)",
        "land_area": "જમીન વિસ્તાર (એકર)",
        "yield": "અપેક્ષિત ઉત્પાદન (ક્વિન્ટલ/એકર)",
        "nitrogen": "નાઇટ્રોજન",
        "phosphorus": "ફોસ્ફરસ",
        "potassium": "પોટેશિયમ",
        "magnesium": "મેગ્નેશિયમ",
        "zinc": "ઝિંક",
        "hardness": "માટીની કઠિનતા (%)",
        "profit_form": "કસ્ટમ નફો વિશ્લેષણ",
        "select_crop": "નફા વિશ્લેષણ માટે પાક પસંદ કરો",
        "cost_per_quintal": "દર ક્વિન્ટલ દીઠ ઇનપુટ ખર્ચ (₹)",
        "market_price_input": "દર ક્વિન્ટલ દીઠ વેચાણ ભાવ (₹)",
        "calc_profit": "નફો/નુકસાન ગણો",
        "acidic": "⚠ અમ્લીય માટી",
        "alkaline": "⚠ ક્ષારીય માટી",
        "good_ph": "✅ pH સારું છે",
        "upload_leaf": "પાન અપલોડ કરો",
        "pest_msg": "⚠ જીવાત માટે એઆઈ મોડેલ ટૂંક સમયમાં આવી રહ્યું છે.",
        "feedback_input": "પ્રતિસાદ દાખલ કરો",
        "feedback_btn": "પ્રતિસાદ મોકલો",
        "feedback_success": "✅ તમારા પ્રતિસાદ માટે આભાર!",
        "prediction_failed": "અગાઉથી અંદાજ નિષ્ફળ ગયો",
        "selling_price": "વેચાણ ભાવ",
        "input_cost": "ઇનપુટ ખર્ચ",
        "total_input_cost": "કુલ ઇનપુટ ખર્ચ",
        "revenue": "કુલ આવક",
        "profit_result": "શુદ્ધ નફો",
        "loss_result": "શુદ્ધ નુકસાન",
        "price_col": "વેચાણ ભાવ (₹/ક્વિન્ટલ)",
        "crop_col": "પાક",
        "city": "📍 શહેર",
        "adv_toggle": "Mg/Zn/કઠિનતા સક્રિય કરો",
        "voice_out_toggle": "વોઇસ આઉટપુટ સક્રિય કરો",
        "voice_in_toggle": "વોઇસ ઇનપુટ (STT) સક્રિય કરો",
        "voice_failed": "વોઇસ નિષ્ફળ ગયું",
        "top_crops_voice": "ટોપ પાક",
        "crop_names": {
    "rice": "ચોખા", "wheat": "ગહું", "maize": "મકાઇ",
    "sugarcane": "ઈખ", "cotton": "કપાસ", 
    "potato": "બટાકા", "Soybean": "સોયાબીન", "adzuki beans": "અડદની ફળી", 
    "apple": "સફરજન", "banana": "કેળું", "black gram": "ઉડદ", 
    "chickpea": "ચણા", "coconut": "નાળિયેર", "coffee": "કોફી", 
    "grapes": "દ્રાક્ષ", "ground nut": "શીંગદાણા", "jute": "જૂટ", 
    "kidney beans": "રાજમા", "lentil": "મસૂર", "mango": "કેરી", 
    "millet": "બાજરી", "moth beans": "મઠ બીન્સ", "mung bean": "મગ", 
    "muskmelon": "ખરબુજ", "orange": "નારંગી", "papaya": "પપૈયું", 
    "peas": "વાલ", "pigeon peas": "તુવર", "pomegranate": "દાડમ", 
    "rubber": "રબર", "tea": "ચા", "tobacco": "તંબાકુ", 
    "watermelon": "તરબૂચ"
},
        "desc": {
            "rice": "ધાન — મુખ્ય અનાજ પાક",
            "wheat": "ગહું — અનાજ પાક",
            "maize": "મકાઈ — બહુઉપયોગી પાક",
            "sugarcane": "ઇખ — રોકડ પાક",
            "cotton": "કપાસ — રેસા પાક",
            "potato": "બટાટા — કંદ પાક",
            "Soybean": "સોયાબીન — તેલબિયાં પાક"
        }
    }


}

# -----------------------
# Language Selector
# -----------------------
lang = st.sidebar.selectbox("🌐 Language / भाषा / ਭਾਸ਼ਾ / ভাষা / ભાષા", ["English","हिन्दी","ਪੰਜਾਬੀ","বাংলা","ગુજરાતી"])
t = TRANSLATIONS[lang]

# Crop details (cost/price defaults; names & desc come from t)
# -----------------------
crop_details = {
    "rice": {"image":"rice.jpg"},
    "wheat":{"image":"wheat.jpg"},
    "maize":{"image":"maize.jpg",},
    "sugarcane":{"image":"sugarcane.jpg"},
    "cotton":{"image":"cotton.jpg"},
    "potato":{"image":"potato.jpg"},
    "Soybean":{"image":"s.jpg"},
    "peas":{"image":"peas.jpg"}
}
def display_name(key):
    return t["crop_names"].get(key, key.capitalize())


# Sidebar options
city = st.sidebar.text_input(t["city"], "Panipat")
enable_adv = st.sidebar.checkbox(t["adv_toggle"])
enable_voice = st.sidebar.checkbox(t["voice_out_toggle"])
if SR_AVAILABLE:
    enable_voice_input = st.sidebar.checkbox(t["voice_in_toggle"])
else:
    enable_voice_input = False
st.title(t["title"])
st.write(t["subtitle"])

# -----------------------
# Input form
# -----------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        soil_moisture = st.slider(t["soil_moisture"], 0, 100, 35)
        soil_ph = st.slider(t["soil_ph"], 0.0, 14.0, 6.5)
        temp = st.slider(t["temp"], -5, 55, 28)
    with col2:
        humidity = st.slider(t["humidity"], 0, 100, 70)
        P = st.slider(t["phosphorus"], 0, 150, 50)
        N = st.slider(t["nitrogen"], 0, 150, 50)

    col1, col2 = st.columns(2)
    with col1:
        K = st.slider(t["potassium"], 0, 200, 50)
        soil_type = st.selectbox(t["soil_type"], ["Loamy Soil","Sandy Soil","Clay Soil","Alluvial Soil","Black Soil","Red Soil"])
    with col2:
        land_area = st.number_input(t["land_area"], 1, 1000, 3)
        yield_est = st.number_input(t["yield"], 1, 100, 20)

    if enable_adv:
        col1, col2 = st.columns(2)
        with col1:
            Mg = st.slider(t["magnesium"], 0, 200, 50)
        with col2:
            Zn = st.slider(t["zinc"], 0, 50, 10)
        soil_hardness = st.slider(t["hardness"], 0, 100, 35)
    else:
        Mg = None; Zn = None; soil_hardness = None

    submitted = st.form_submit_button(t["predict"])

# -----------------------
# If user requested live weather, fetch and store in session_state so input boxes update
# -----------------------
if city:
    w = fetch_weather_openweathermap(city)
    if w:
        st.session_state["weather_temp"] = w["main"]["temp"]
        st.session_state["weather_hum"] = w["main"]["humidity"]
        # OpenWeather gives precipitation in different fields; approximate as 0 if missing
        rain = 0
        if "rain" in w and isinstance(w["rain"], dict):
            # '1h' or '3h'
            rain = w["rain"].get("1h", w["rain"].get("3h", 0))
        st.session_state["weather_rain"] = rain

# -----------------------
# On submit -> prepare features expected by model and predict
# -----------------------
if submitted:
    rainfall = st.session_state.get("weather_rain", 0.0)
    carbon = 1.5  # default if not collected from UI

    sample = {
        "Temperature": float(temp),
        "Humidity": float(humidity),
        "Rainfall": float(rainfall),
        "PH": float(soil_ph),
        "Nitrogen": float(N),
        "Phosphorous": float(P),
        "Potassium": float(K),
        "Carbon": float(carbon),
        "Soil": str(soil_type)
    }
    df_sample = pd.DataFrame([sample])

    # Validate model availability
    if model is None or le is None:
        st.error("Model files not found. Please run training (src/train_model.py) and ensure 'models/crop_model_pipeline.pkl' and 'models/label_encoder.pkl' exist.")
        st.stop()

    pred, proba, classes = predict_with_model(model, df_sample)
    if pred is None:
        st.error(t["prediction_failed"])
        st.stop()

    # Determine class names (use label encoder if available)
    try:
        # If model returns encoded label index, use label encoder
        if isinstance(pred, (int, np.integer)) and le is not None:
            crop_name = le.inverse_transform([pred])[0]
        else:
            # If model returns string labels directly
            crop_name = str(pred)
    except Exception:
        crop_name = str(pred)

    # Build probability DataFrame if proba available
    if proba is not None and classes is not None:
        # classes list may be encoded or string — convert to lowercase keys
        classes_list = [str(c) for c in classes]
        probs = proba[0]
        df_prob = pd.DataFrame({"Crop": classes_list, "Probability": probs})
        df_prob = df_prob.sort_values("Probability", ascending=False).reset_index(drop=True)
        top3 = df_prob.head(3)
    else:
        top3 = pd.DataFrame([{"Crop": crop_name, "Probability": 1.0}])

    # Save into session for later display
    st.session_state["top3"] = top3
    st.session_state["best_crop"] = crop_name

# -----------------------
# Post-prediction display
# -----------------------
if "top3" in st.session_state:
    top3 = st.session_state["top3"]
    best_crop = st.session_state["best_crop"]

    st.header(t["recommended"])
    # Localize display names if you have translations for crop keys
    def localized_name(name_key):
        nk = str(name_key).lower()
        # try to find in t["crop_names"]
        for k, v in t.get("crop_names", {}).items():
            if k.lower() == nk or v.lower() == nk:
                return v
        return name_key

    # Best crop box
    best_name = localized_name(best_crop)
    st.markdown(f"<div style='font-size:28px; font-weight:bold; color:darkgreen'>🌟 {best_name} — {t['best']}</div>", unsafe_allow_html=True)

    # Show top3 with probabilities
    st.subheader("Top options")
    st.table(top3.assign(Probability=lambda d: (d["Probability"]*100).round(1).astype(str) + "%").rename(columns={"Crop": "Crop", "Probability": "Probability"}))

    # Voice output (optional)
    if enable_voice:
        voice_items = []
        for _, r in top3.iterrows():
            voice_items.append(f"{localized_name(r['Crop'])} {r['Probability']*100:.1f} percent")
        txt = f"{t['top_crops_voice']}: " + ", ".join(voice_items)
        lang_map = {"English": "en", "हिन्दी": "hi", "ਪੰਜਾਬੀ": "pa"}
        try:
            mp3 = gTTS(txt, lang=lang_map.get(lang, "en"))
            buf = io.BytesIO()
            mp3.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf, format="audio/mp3")
        except Exception:
            st.error(t["voice_failed"])

    # Soil health and weather summary (your existing blocks can follow here)
    st.header(t["soil_health"])
    if soil_ph < 5.5:
        st.warning(t["acidic"])
    elif soil_ph > 8:
        st.warning(t["alkaline"])
    else:
        st.success(t["good_ph"])

    st.header(t["weather"])
    w = fetch_weather_openweathermap(city) if city else None
    if w:
        st.write(f"🌡 {w['main']['temp']}°C, 💧{w['main']['humidity']}%, ☁ {w['weather'][0]['description']}")
    else:
        st.write("Weather not available")
    st.header(t["pest"])
    uploaded = st.file_uploader(t["upload_leaf"], type=["jpg","png"])
    if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True)
            st.warning(t["pest_msg"])

        # -----------------------
        # Market Prices
        # -----------------------
        # Feedback
        # -----------------------
st.header(t["feedback"])
fb = st.text_area(t["feedback_input"])
if st.button(t["feedback_btn"]):
            st.success(t["feedback_success"])

        # -----------------------
        # Voice Output (Top crops)
        # -----------------------
if enable_voice and prob_arr is not None:
            top = top3.copy()
            # Build localized voice stringu
            voice_items = []
            for _, r in top.iterrows():
                k = r["CropKey"]
                voice_items.append(f"{display_name(k)} {r['%']:.1f}%")
            txt = f"{t['top_crops_voice']}: " + ", ".join(voice_items)

            lang_map = {"English": "en", "हिन्दी": "hi", "ਪੰਜਾਬੀ": "pa"}
            try:
                mp3 = gTTS(txt, lang=lang_map.get(lang, "en"))
                buf = io.BytesIO()
                mp3.write_to_fp(buf)
                buf.seek(0)
                st.audio(buf, format="audio/mp3")
            except Exception:
                st.error(t["voice_failed"])
                # Display fetched weather info
            st.success(
            f"🌤 Current weather in {city}: "
            f"{w['main']['temp']}°C | Humidity: {w['main']['humidity']}%"
        )
else:
        st.warning("")

# -----------------------


 