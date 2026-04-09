import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf, numpy as np, os, json, cv2
from PIL import Image
from modules.leaf_detector      import is_leaf
from modules.severity_estimator import estimate_severity
from modules.image_quality      import check_image_quality
from modules.knowledge_base     import get_chatbot_response, get_disease_info, get_severity_advice
from modules.voice_engine       import speak_text, stop_speaking, build_voice_text, check_hindi_voice_installed
from modules.leaf_age_detector  import detect_leaf_age, AGE_UI
import modules.db_manager as db

st.set_page_config(page_title="Plant Doctor AI", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{font-family:'Inter',-apple-system,sans-serif!important;background:#212121!important;color:#ececec!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0!important;max-width:100%!important}
[data-testid="stSidebar"]{background:#171717!important;border:none!important;width:260px!important;min-width:260px!important;max-width:260px!important}
[data-testid="stSidebar"]>div{padding:0!important}
div.stButton>button{background:transparent!important;color:#c2c2c2!important;border:none!important;border-radius:8px!important;padding:9px 10px!important;font-size:14px!important;font-weight:400!important;text-align:left!important;width:100%!important;font-family:'Inter',sans-serif!important;transition:background .12s!important}
div.stButton>button:hover{background:#2a2a2a!important;color:#ececec!important}
div.stButton>button:focus{outline:none!important;box-shadow:none!important}
div.stSelectbox>label{display:none!important}
div.stSelectbox>div>div{background:#2a2a2a!important;border:1px solid #333!important;border-radius:8px!important;color:#ececec!important;font-size:13px!important}
div.stFileUploader>label{display:none!important}
div.stFileUploader section{background:#2a2a2a!important;border:1px dashed #3a3a3a!important;border-radius:10px!important;padding:8px!important}
div[data-testid="stExpander"]{background:#252525!important;border:1px solid #333!important;border-radius:10px!important;margin-bottom:4px!important}
div[data-testid="stExpander"] summary{font-size:13px!important;color:#676767!important}
/* Hide the message bus form completely */
div[data-testid="stForm"]{position:fixed!important;bottom:-9999px!important;left:-9999px!important;opacity:0!important;pointer-events:none!important;z-index:-999!important}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:#333;border-radius:4px}

.topbar{padding:10px 20px;display:flex;align-items:center;justify-content:center;border-bottom:1px solid #2a2a2a;background:#212121;position:sticky;top:0;z-index:50}
.topbar-title{font-size:15px;font-weight:600;color:#ececec;display:flex;align-items:center;gap:5px}
.topbar-sub{font-size:12px;color:#555;font-weight:400}
.topbar-r{position:absolute;right:20px}
.live-badge{display:flex;align-items:center;gap:5px;font-size:12px;color:#19c37d}
.ldot{width:6px;height:6px;border-radius:50%;background:#19c37d;animation:blink 1.4s infinite;display:inline-block}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

.chat{max-width:680px;margin:0 auto;padding:24px 16px 220px}
.mrow{display:flex;gap:14px;margin-bottom:28px;align-items:flex-start}
.mrow.u{flex-direction:row-reverse}
.av{width:28px;height:28px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;margin-top:2px}
.av.b{background:#19c37d;color:#fff}
.av.u{background:#ab68ff;color:#fff}
.mb{flex:1;min-width:0}
.mn{font-size:13px;font-weight:600;color:#ececec;margin-bottom:6px}
.utxt{font-size:15px;line-height:1.65;color:#ececec}
.btxt{font-size:15px;line-height:1.8;color:#d1d5db}
.vrow{display:flex;gap:5px;margin-top:8px}
.vb{padding:3px 10px;border-radius:20px;border:1px solid #2a2a2a;background:transparent;font-size:12px;color:#555;cursor:pointer;font-family:'Inter',sans-serif;transition:all .1s}
.vb:hover{border-color:#444;color:#d1d5db}

.dc{border:1px solid #2a2a2a;border-radius:16px;overflow:hidden;margin-top:4px}
.dc-h{padding:16px 18px 12px;background:#1c1c1c}
.dc-n{font-size:17px;font-weight:600;color:#fff;margin-bottom:3px}
.dc-m{font-size:12px;color:#555;display:flex;gap:14px;margin-bottom:10px}
.dc-pb{height:2px;background:#2a2a2a;border-radius:1px;overflow:hidden;margin-bottom:10px}
.dc-pf{height:100%;background:#19c37d;border-radius:1px}
.sv{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:500}
.sv-L{background:#051a0d;color:#19c37d;border:1px solid #19c37d25}
.sv-M{background:#1a1000;color:#f59e0b;border:1px solid #f59e0b25}
.sv-H{background:#1a0505;color:#ef4444;border:1px solid #ef444425}
.dc-adv{font-size:13px;color:#555;margin-top:8px;line-height:1.5}
.dc-g{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #2a2a2a}
.dc-c{padding:11px 16px;border-right:1px solid #2a2a2a;border-top:1px solid #2a2a2a;background:#171717}
.dc-c:nth-child(2n){border-right:none}
.dc-l{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:#19c37d;font-weight:600;margin-bottom:3px}
.dc-v{font-size:13px;color:#8e8ea0;line-height:1.45}

.ac{border:1px solid #2a2a2a;border-radius:16px;overflow:hidden;margin-top:8px}
.ac-h{padding:14px 18px 10px;background:#1c1c1c}
.ac-sl{font-size:10px;text-transform:uppercase;letter-spacing:.7px;color:#424242;font-weight:500;margin-bottom:5px}
.ac-n{font-size:15px;font-weight:600;margin-bottom:2px}
.ac-r{font-size:12px;color:#424242;margin-bottom:8px}
.ac-b{height:2px;background:#2a2a2a;border-radius:1px;overflow:hidden}
.ac-g{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #2a2a2a}
.ac-c{padding:10px 14px;border-right:1px solid #2a2a2a;border-top:1px solid #2a2a2a;background:#171717}
.ac-c:nth-child(2n){border-right:none}
.ac-cl{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:#333;font-weight:500;margin-bottom:2px}
.ac-cv{font-size:12px;color:#555;line-height:1.4}

.cc{background:#071a10;border:1px solid #19c37d18;border-radius:16px;padding:16px 18px;margin-top:4px}
.cc-t{display:flex;align-items:center;gap:7px;margin-bottom:8px}
.cc-d{width:7px;height:7px;border-radius:50%;background:#19c37d}
.cc-l{font-size:12px;font-weight:600;color:#19c37d}
.cc-b{font-size:15px;color:#8e8ea0;line-height:1.8}
.errc{background:#1a0505;border:1px solid #ef444420;border-radius:10px;padding:10px 14px;font-size:14px;color:#f87171;margin-top:4px}

.wlc{display:flex;align-items:center;justify-content:center;padding:120px 20px 20px}
.wlc h1{font-size:32px;font-weight:600;color:#ececec;letter-spacing:-.5px;text-align:center}

.ibar{position:fixed;bottom:0;left:260px;right:0;padding:12px 24px 20px;background:linear-gradient(to top,#212121 80%,transparent);z-index:100}
.ibar-in{max-width:680px;margin:0 auto}
.ifoot{font-size:12px;color:#333;text-align:center;margin-top:8px}
</style>
""", unsafe_allow_html=True)

T={
 "en":{"title":"Plant Doctor AI","you":"You","bot":"Plant Doctor",
  "wh":"What's troubling your crops today?",
  "ph":"Ask anything...","newc":"New chat",
  "hlth":"Plant looks healthy","nodis":"No disease detected",
  "conf":"Confidence","inf":"Infected","cau":"Cause","prv":"Prevention",
  "chm":"Chemical","org":"Organic","orig":"Original","infa":"Infection area",
  "fup":"Ask me: organic treatment? how to prevent? what caused this?",
  "anal":"Analysing...","lang":"Language","rec":"Your chats","priv":"Data stays on your device",
  "live":"Voice call active","lc":"en-IN","mls":"Listening...","mch":"Use Chrome for voice",
  "btn_call":"📞 Call","btn_stop":"⏹ Stop","btn_attach":"📎 Attach","btn_send":"↑",
  "cg":"Hello! I'm your Plant Doctor AI.\n\nI can help with:\n• Disease diagnosis from leaf photos\n• Treatment advice — organic & chemical\n• Crop care: wheat, rice, cotton, mustard\n• Pest management & farming tips\n\nDescribe your problem or upload a leaf photo.",
 },
 "hi":{"title":"पौधा डॉक्टर AI","you":"आप","bot":"पौधा डॉक्टर",
  "wh":"आज आपकी फसल में क्या समस्या है?",
  "ph":"कुछ भी पूछें...","newc":"नई बातचीत",
  "hlth":"पौधा स्वस्थ है","nodis":"कोई बीमारी नहीं",
  "conf":"सटीकता","inf":"प्रभावित","cau":"कारण","prv":"सावधानी",
  "chm":"रासायनिक","org":"जैविक","orig":"मूल फोटो","infa":"संक्रमित क्षेत्र",
  "fup":"पूछें: जैविक उपचार? कैसे रोकें? यह क्यों हुआ?",
  "anal":"जांच हो रही है...","lang":"भाषा","rec":"आपकी बातचीत","priv":"डेटा आपके डिवाइस पर",
  "live":"वॉयस कॉल सक्रिय","lc":"hi-IN","mls":"सुन रहा है...","mch":"Voice के लिए Chrome उपयोग करें",
  "btn_call":"📞 कॉल","btn_stop":"⏹ रोकें","btn_attach":"📎 फोटो","btn_send":"↑",
  "cg":"Namaste! Mein aapka Plant Doctor AI hoon.\n\nMein madad kar sakta hoon:\n• Paudhon ki bimari — photo se pata lagana\n• Ilaaj — jaivik aur rasayanik\n• Fasal dekhbhaal\n• Kheti ke sawaal\n\nSamasya batayein ya patte ki photo upload karein.",
 },
 "pa":{"title":"ਪੌਦਾ ਡਾਕਟਰ AI","you":"ਤੁਸੀਂ","bot":"ਪੌਦਾ ਡਾਕਟਰ",
  "wh":"ਅੱਜ ਤੁਹਾਡੀ ਫਸਲ ਵਿੱਚ ਕੀ ਸਮੱਸਿਆ ਹੈ?",
  "ph":"ਕੁਝ ਵੀ ਪੁੱਛੋ...","newc":"ਨਵੀਂ ਗੱਲਬਾਤ",
  "hlth":"ਪੌਦਾ ਸਿਹਤਮੰਦ ਹੈ","nodis":"ਕੋਈ ਬਿਮਾਰੀ ਨਹੀਂ",
  "conf":"ਸਹੀਅਤ","inf":"ਪ੍ਰਭਾਵਿਤ","cau":"ਕਾਰਨ","prv":"ਸਾਵਧਾਨੀ",
  "chm":"ਰਾਸਾਇਣਿਕ","org":"ਜੈਵਿਕ","orig":"ਅਸਲ","infa":"ਸੰਕ੍ਰਮਿਤ ਖੇਤਰ",
  "fup":"ਪੁੱਛੋ: ਜੈਵਿਕ ਇਲਾਜ? ਕਿਵੇਂ ਰੋਕੀਏ? ਇਹ ਕਿਉਂ ਹੋਇਆ?",
  "anal":"ਜਾਂਚ ਹੋ ਰਹੀ ਹੈ...","lang":"ਭਾਸ਼ਾ","rec":"ਤੁਹਾਡੀ ਗੱਲਬਾਤ","priv":"ਡੇਟਾ ਤੁਹਾਡੇ ਡਿਵਾਈਸ ਤੇ",
  "live":"ਵੌਇਸ ਕਾਲ ਸਕਿਰਿਆ","lc":"pa-IN","mls":"ਸੁਣ ਰਿਹਾ ਹੈ...","mch":"Voice ਲਈ Chrome ਵਰਤੋ",
  "btn_call":"📞 ਕਾਲ","btn_stop":"⏹ ਰੋਕੋ","btn_attach":"📎 ਫੋਟੋ","btn_send":"↑",
  "cg":"Sat Sri Akal! Mein tauhada Plant Doctor AI haan.\n\nMein madad kar sakda haan:\n• Paudhiaan di bimari — photo to pata lagana\n• Ilaaj — jaivik te rasayanik\n• Fasal dekhbhaal\n• Kheti de sawaal\n\nSamasya dasoo ya patte di photo upload karo.",
 },
}
SEV={
 "en":{"LOW":"Low severity","MEDIUM":"Moderate severity","HIGH":"Severe — act now"},
 "hi":{"LOW":"हल्की","MEDIUM":"मध्यम","HIGH":"गंभीर — तुरंत करें"},
 "pa":{"LOW":"ਹਲਕੀ","MEDIUM":"ਦਰਮਿਆਨੀ","HIGH":"ਗੰਭੀਰ — ਤੁਰੰਤ ਕਰੋ"},
}
LMAP={"English":"en","Hindi":"hi","Punjabi":"pa"}

@st.cache_resource
def load_model():
    p="model/plant_disease_model.h5"
    return tf.keras.models.load_model(p) if os.path.exists(p) else None
@st.cache_resource
def load_classes():
    j="model/class_names.json"
    if os.path.exists(j):
        with open(j) as f: d=json.load(f)
        return {int(k):v for k,v in d.items()}
    dp="dataset/PlantVillage"
    if os.path.exists(dp):
        ns=sorted([x for x in os.listdir(dp) if os.path.isdir(os.path.join(dp,x))])
        return {i:n for i,n in enumerate(ns)}
    return {}

mdl=load_model(); cmap=load_classes()

def predict(img):
    if not mdl: return "Unknown",0.0
    # Resizing exactly to the model's expected shape
    img_resized = img.convert("RGB").resize((224, 224))
    # Normalizing image pixels between -1 and 1
    a = np.array(img_resized, dtype=np.float32) / 127.5 - 1.0
    # Adding batch dimension with np.expand_dims to create 4D tensor
    a = np.expand_dims(a, axis=0)
    
    p = mdl.predict(a, verbose=0)
    i = int(np.argmax(p))
    return cmap.get(i, "Unknown").replace("__", " ").replace("_", " "), float(np.max(p))

def highlight(img):
    a=np.array(img.convert("RGB")); b=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
    h=cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
    cm=cv2.inRange(h,np.array([5,40,20]),np.array([25,255,200]))
    for lo,hi in [([160,40,20],[180,255,180]),([20,60,60],[35,255,255])]:
        cm=cv2.bitwise_or(cm,cv2.inRange(h,np.array(lo),np.array(hi)))
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cm=cv2.morphologyEx(cv2.morphologyEx(cm,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k)
    cs,_=cv2.findContours(cm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); out=a.copy()
    for c in cs:
        if cv2.contourArea(c)>600:
            x,y,w,h2=cv2.boundingRect(c); ov=out.copy()
            cv2.rectangle(ov,(x,y),(x+w,y+h2),(220,60,60),-1)
            out=cv2.addWeighted(out,.78,ov,.22,0)
            cv2.rectangle(out,(x,y),(x+w,y+h2),(200,50,50),2)
    return Image.fromarray(out)

for k,v in [("msgs",[]),("hist",[]),("last_dis",None),("lng","English"),
            ("cid",0),("proc",None),("call",False),
            ("pending_text",""),("pending_action",""),("app_mode","Chat"),
            ("selected_plant",None)]:
    if k not in st.session_state: st.session_state[k]=v

def add(r,c,t="text",**kw): st.session_state.msgs.append({"r":r,"c":c,"t":t,**kw})

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding:14px 10px 8px;display:flex;align-items:center;gap:10px"><div style="width:30px;height:30px;border-radius:8px;background:#19c37d;display:flex;align-items:center;justify-content:center;font-size:15px;color:#fff;flex-shrink:0">🌿</div><span style="font-size:14px;font-weight:500;color:#ececec">Plant Doctor AI</span></div>',unsafe_allow_html=True)

    sel=st.selectbox("L",["English","Hindi","Punjabi"],
                     index=["English","Hindi","Punjabi"].index(st.session_state.lng))
    if sel!=st.session_state.lng: st.session_state.lng=sel; st.rerun()
    lk=LMAP[st.session_state.lng]; tx=T[lk]

    st.markdown(f'<div style="padding:2px 10px 3px;font-size:12px;color:#555;font-weight:500">{tx["lang"]}</div>',unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:#2a2a2a;margin:4px 0"></div>',unsafe_allow_html=True)

    st.markdown('<div style="padding:4px 10px;font-size:12px;color:#555">Views</div>',unsafe_allow_html=True)
    if st.button("💬 Chat Mode", key="btn_chat"):
        st.session_state.app_mode = "Chat"; st.rerun()
    if st.button("📈 Tracker", key="btn_track"):
        st.session_state.app_mode = "Progress Tracker"; st.rerun()
        
    st.markdown('<div style="height:1px;background:#2a2a2a;margin:4px 0"></div>',unsafe_allow_html=True)

    if st.session_state.selected_plant:
        st.markdown(f'<div style="padding:4px 10px;font-size:11px;color:#19c37d">Context: {st.session_state.selected_plant}</div>',unsafe_allow_html=True)

    if st.button(f"✏  {tx['newc']}"):
        if st.session_state.msgs:
            f=next((m["c"] for m in st.session_state.msgs if m["r"]=="u"),"Chat")
            p=(str(f)[:36]+"…") if len(str(f))>36 else str(f)
            st.session_state.hist.insert(0,{"id":st.session_state.cid,"p":p,"ms":st.session_state.msgs.copy()})
        st.session_state.msgs=[]; st.session_state.last_dis=None
        st.session_state.proc=None; st.session_state.call=False
        stop_speaking(); st.session_state.cid+=1; st.rerun()

    if st.session_state.hist:
        st.markdown(f'<div style="padding:8px 10px 3px;font-size:12px;color:#555">{tx["rec"]}</div>',unsafe_allow_html=True)
        for item in st.session_state.hist[:10]:
            if st.button(item["p"][:38],key=f"h{item['id']}"):
                st.session_state.msgs=item["ms"].copy(); st.session_state.last_dis=None
                st.session_state.proc=None; st.session_state.call=False; st.rerun()
    else:
        st.markdown('<div style="padding:6px 12px;font-size:13px;color:#2a2a2a">No chats yet</div>',unsafe_allow_html=True)

    if lk!="en" and not check_hindi_voice_installed():
        st.markdown(f'<div style="margin:8px;padding:8px;background:#2a1800;border:1px solid #554400;border-radius:8px;font-size:11px;color:#888">💡 For {sel} voice: Settings → Language → Add {sel} → Speech</div>',unsafe_allow_html=True)
    st.markdown(f'<div style="padding:8px 12px;font-size:11px;color:#2a2a2a;border-top:1px solid #2a2a2a;margin-top:8px">🔒 {tx["priv"]}</div>',unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
live_html=(f'<div class="live-badge"><span class="ldot"></span>{tx["live"]}</div>'
           if st.session_state.call else f'<span style="font-size:12px;color:#333">{sel}</span>')
st.markdown(f'<div class="topbar"><div class="topbar-title">{tx["title"]} <span class="topbar-sub">▾</span></div><div class="topbar-r">{live_html}</div></div>',unsafe_allow_html=True)

# ── VOICE BUTTONS ─────────────────────────────────────────────────────────────
def vbtns(idx,txt):
    v1,v2,_=st.columns([1,1,10])
    with v1:
        if st.button("🔊",key=f"sp{idx}"): speak_text(txt[:400],sel)
    with v2:
        if st.button("⏹",key=f"st{idx}"): stop_speaking()

# ── PROGRESS TRACKER VIEW ─────────────────────────────────────────────────────
if st.session_state.app_mode == "Progress Tracker":
    st.markdown('<div style="padding: 20px 40px; max-width: 900px; margin: 0 auto;">', unsafe_allow_html=True)
    st.markdown("## 📈 Plant Progress Tracker", unsafe_allow_html=True)
    st.markdown("Monitor crop health and record treatments over time.", unsafe_allow_html=True)
    
    plants = db.get_all_plant_profiles()
    plant_names = [p["name"] for p in plants]
    
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("### 🌿 Select Plant Dashboard")
        curr_idx = 0
        if st.session_state.selected_plant and st.session_state.selected_plant in plant_names:
            curr_idx = plant_names.index(st.session_state.selected_plant) + 1
        
        selected = st.selectbox("Choose a plant to view history", ["-- Context: No Plant Set --"] + plant_names, index=curr_idx)
        
        new_selection = selected if selected != "-- Context: No Plant Set --" else None
        if new_selection != st.session_state.selected_plant:
            st.session_state.selected_plant = new_selection
            st.rerun()
            
    with col2:
        st.markdown("### ➕ Register New Plant")
        with st.expander("Register a new field or plant"):
            with st.form("new_plant_form"):
                n_name = st.text_input("Identify Name (e.g. Frontyard Tomato View)")
                n_crop = st.text_input("Crop/Seed Type (e.g. Tomato)")
                n_date = st.date_input("Sown Date")
                if st.form_submit_button("Save Plant"):
                    if n_name:
                        db.add_plant_profile(n_name, n_crop, sown_date=str(n_date))
                        st.session_state.selected_plant = n_name
                        st.rerun()

    if st.session_state.selected_plant:
        db_plant = next((p for p in plants if p["name"] == st.session_state.selected_plant), None)
        if db_plant:
            st.markdown("---")
            st.markdown(f"### Profile: **{db_plant['name']}** ({db_plant['crop_type']}) &nbsp;&nbsp;|&nbsp;&nbsp; Sown: {db_plant['sown_date']}", unsafe_allow_html=True)
            
            c1, c2 = st.columns([1.5, 1], gap="medium")
            with c1:
                st.markdown("#### 🔬 AI Scan History")
                diags = db.get_all_diagnoses(limit=100) 
                p_diags = [d for d in diags if d.get("plant_name") == db_plant['name']]
                
                if not p_diags:
                    st.info("No AI scans recorded for this plant yet. Select this plant in the sidebar, switch to Chat mode, and upload a leaf image to log a scan.")
                else:
                    for d in p_diags:
                        sev = str(d.get("severity", "LOW"))
                        sev_color = "#ef4444" if sev == "HIGH" else "#f59e0b" if sev == "MEDIUM" else "#19c37d"
                        is_h = "healthy" in str(d["disease"]).lower()
                        d_name = d["disease"].replace("_", " ") if hasattr(d["disease"], "replace") else d["disease"]
                        st.markdown(f'''
                        <div style="background:#171717; padding: 10px 14px; border-radius: 8px; border: 1px solid #333; margin-bottom: 8px;">
                            <div style="font-size: 11px; color: #888; margin-bottom: 3px;">{d['timestamp']}</div>
                            <div style="font-size: 15px; font-weight: 600; color: {'#19c37d' if is_h else '#ececec'}">{d_name}</div>
                            {"" if is_h else f'<div style="font-size: 13px; color: {sev_color}; margin-top: 3px;"><i>{sev} Severity</i> &nbsp;•&nbsp; {round(d["confidence"]*100)}% Match</div>'}
                        </div>
                        ''', unsafe_allow_html=True)
            
            with c2:
                st.markdown("#### 💊 Treatment Logs")
                treatments = db.get_treatment_log(limit=100)
                p_treatments = [t for t in treatments if t.get("plant_name") == db_plant['name']]
                
                with st.expander("➕ Log New Treatment", expanded=not bool(p_treatments)):
                    with st.form("treat_form", clear_on_submit=True):
                        tr_name = st.text_input("Treatment Used (Product Name/Action)")
                        tr_type = st.selectbox("Type", ["Chemical", "Organic", "Manual", "Other"])
                        tr_res = st.selectbox("Result Observation", ["Pending", "Improved", "No Change", "Worsened"])
                        if st.form_submit_button("Log Treatment"):
                            if tr_name:
                                db.log_treatment(db_plant['name'], tr_type, tr_name, result=tr_res)
                                st.rerun()
                                
                if not p_treatments:
                    st.markdown("<div style='font-size:13px; color:#888; margin-top:10px'>No treatments applied yet.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    for t in p_treatments:
                        res = t.get("result", "")
                        res_c = "#19c37d" if res == "Improved" else "#ef4444" if res == "Worsened" else "#f59e0b" if res == "Pending" else "#888"
                        st.markdown(f'''
                        <div style="background:#071a10; padding: 10px 14px; border-radius: 8px; border: 1px solid #19c37d30; margin-bottom: 8px; position: relative;">
                            <div style="font-size: 11px; color: #888; margin-bottom: 3px;">{t['timestamp']}</div>
                            <div style="font-size: 14px; color: #ececec; font-weight: 500;">{t['treatment_used']} <span style="font-weight: 400; color: #aaa; font-size: 12px;">({t['treatment_type']})</span></div>
                            <div style="font-size: 12px; color: {res_c}; margin-top: 4px; font-weight: 500;">Result: {res}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ── MESSAGES ─────────────────────────────────────────────────────────────────
if not st.session_state.msgs:
    st.markdown(f'<div class="wlc"><h1>{tx["wh"]}</h1></div>',unsafe_allow_html=True)
else:
    st.markdown('<div class="chat">',unsafe_allow_html=True)
    for idx,msg in enumerate(st.session_state.msgs):
        r=msg["r"]; t=msg.get("t","text"); c=msg.get("c","")
        if r=="u":
            avi=tx["you"][0]
            st.markdown(f'<div class="mrow u"><div class="av u">{avi}</div><div class="mb"><div class="mn">{tx["you"]}</div>',unsafe_allow_html=True)
            if t=="img": st.image(msg["img"],width=220)
            elif t=="cs": st.markdown('<span style="font-size:14px;color:#19c37d">📞 Voice call started</span>',unsafe_allow_html=True)
            else: st.markdown(f'<div class="utxt">{str(c).replace(chr(10),"<br>")}</div>',unsafe_allow_html=True)
            st.markdown('</div></div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="mrow"><div class="av b">🌿</div><div class="mb"><div class="mn">{tx["bot"]}</div>',unsafe_allow_html=True)
            if t=="dx":
                d=msg["dis"]; cf=msg["conf"]; sv=msg["sev"]
                inf=msg.get("info",{}); age=msg.get("age"); sl=SEV[lk]
                cp=round(cf*100); ip=sv["infected_percent"]; is_h="healthy" in d.lower()
                sc={"LOW":"sv-L","MEDIUM":"sv-M","HIGH":"sv-H"}[sv["severity"]]
                if is_h:
                    st.markdown(f'<div class="dc"><div class="dc-h"><div class="dc-n">✅ {d}</div><div class="dc-m"><span>{tx["hlth"]}</span></div><span class="sv sv-L">{tx["nodis"]}</span></div></div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="dc"><div class="dc-h"><div class="dc-n">{d}</div><div class="dc-m"><span>{tx["conf"]}: {cp}%</span><span>{tx["inf"]}: {ip}%</span></div><div class="dc-pb"><div class="dc-pf" style="width:{cp}%"></div></div><span class="sv {sc}">{sl[sv["severity"]]}</span><div class="dc-adv">{sv["advice"]}</div></div>',unsafe_allow_html=True)
                    if inf:
                        st.markdown(f'<div class="dc-g"><div class="dc-c"><div class="dc-l">{tx["cau"]}</div><div class="dc-v">{inf.get("cause","—")}</div></div><div class="dc-c"><div class="dc-l">{tx["prv"]}</div><div class="dc-v">{inf.get("precaution","—")}</div></div><div class="dc-c"><div class="dc-l">{tx["chm"]}</div><div class="dc-v">{inf.get("treatment_chemical","—")}</div></div><div class="dc-c"><div class="dc-l">{tx["org"]}</div><div class="dc-v">{inf.get("treatment_organic","—")}</div></div></div>',unsafe_allow_html=True)
                    st.markdown('</div>',unsafe_allow_html=True)
                if msg.get("hl"):
                    h1,h2=st.columns(2)
                    with h1: st.image(msg["orig"],caption=tx["orig"],use_container_width=True)
                    with h2: st.image(msg["hl"],caption=tx["infa"],use_container_width=True)
                if age:
                    au=AGE_UI.get(sel,AGE_UI["English"]); ap=round(age["age_score"]*100); cp2=round(age["chlorophyll"]*100); ind=age["indicators"]
                    st.markdown(f'<div class="ac"><div class="ac-h"><div class="ac-sl">{au["title"]}</div><div class="ac-n" style="color:{age["color"]}">{age["stage_name"]}</div><div class="ac-r">⏱ {age["age_range"]} · {au["confidence"]}: {round(age["confidence"]*100)}%</div><div class="ac-b"><div style="height:100%;width:{ap}%;background:{age["color"]};border-radius:1px"></div></div></div><div class="ac-g"><div class="ac-c"><div class="ac-cl">{au["desc"]}</div><div class="ac-cv">{age["description"]}</div></div><div class="ac-c"><div class="ac-cl">{au["care"]}</div><div class="ac-cv">{age["care_tip"]}</div></div><div class="ac-c"><div class="ac-cl">{au["chlorophyll"]}</div><div class="ac-cv"><div style="height:2px;background:#2a2a2a;border-radius:1px;overflow:hidden;margin-bottom:3px"><div style="height:100%;width:{cp2}%;background:#19c37d;border-radius:1px"></div></div>{cp2}%</div></div><div class="ac-c"><div class="ac-cl">Yellow · Texture · Veins</div><div class="ac-cv">{round(ind["yellow_ratio"]*100)}% · {round(ind["texture_score"]*100)}% · {round(ind["vein_visibility"]*100)}%</div></div></div></div>',unsafe_allow_html=True)
                te=inf.get("treatment_chemical","") if inf else ""
                vbtns(idx,build_voice_text(d,sv["severity"],sv["advice"],te,sel))
            elif t=="call":
                st.markdown(f'<div class="cc"><div class="cc-t"><div class="cc-d"></div><div class="cc-l">{tx["live"]}</div></div><div class="cc-b">{str(c).replace(chr(10),"<br>")}</div></div>',unsafe_allow_html=True)
                vbtns(idx,c)
            elif t=="err":
                st.markdown(f'<div class="errc">⚠️ {c}</div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="btxt">{str(c).replace(chr(10),"<br>")}</div>',unsafe_allow_html=True)
                if idx==len(st.session_state.msgs)-1: vbtns(idx,c)
            st.markdown('</div></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ── INPUT BAR — single HTML component = one pill ──────────────────────────────
st.markdown('<div class="ibar"><div class="ibar-in">',unsafe_allow_html=True)

# Photo upload (separate, only shown when needed)
with st.expander(tx["btn_attach"],expanded=False):
    uploaded=st.file_uploader("f",type=["jpg","png","jpeg"],
                               key=f"up{st.session_state.cid}{st.session_state.proc or 0}")

# THE ENTIRE INPUT PILL — one HTML component handles text, send, mic, call, stop
lc=tx["lc"]; call_disabled="true" if st.session_state.call else "false"
action_val=st.session_state.pending_action
text_val=st.session_state.pending_text

pill_html=f"""
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',-apple-system,sans-serif}}
.pill{{background:#2f2f2f;border-radius:28px;border:1px solid #3a3a3a;padding:12px 14px 10px 18px;transition:border-color .15s}}
.pill:focus-within{{border-color:#555}}
.row1{{display:flex;align-items:center;gap:10px}}
textarea{{flex:1;background:transparent;border:none;outline:none;resize:none;font-size:15px;color:#ececec;font-family:'Inter',sans-serif;line-height:1.5;max-height:160px;min-height:24px;overflow-y:auto}}
textarea::placeholder{{color:#676767}}
.send{{width:34px;height:34px;border-radius:50%;background:#555;border:none;color:#fff;cursor:pointer;font-size:18px;flex-shrink:0;display:flex;align-items:center;justify-content:center;transition:background .1s}}
.send:hover{{background:#777}}
.send.active{{background:#19c37d}}
.row2{{display:flex;align-items:center;gap:6px;margin-top:8px;border-top:1px solid #2a2a2a;padding-top:8px;flex-wrap:nowrap}}
.abtn{{display:flex;align-items:center;gap:5px;padding:5px 12px;border-radius:20px;border:1px solid #333;background:transparent;font-size:13px;color:#676767;cursor:pointer;font-family:'Inter',sans-serif;transition:all .1s;white-space:nowrap}}
.abtn:hover{{background:#2a2a2a;border-color:#444;color:#ececec}}
.abtn:disabled{{opacity:.4;cursor:not-allowed}}
.abtn.on{{background:#071a10;border-color:#19c37d30;color:#19c37d}}
.mic-btn{{width:34px;height:34px;border-radius:50%;background:#333;border:none;color:#aaa;cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all .15s;margin-left:auto;flex-shrink:0}}
.mic-btn:hover{{background:#444;color:#ececec}}
.mic-btn.on{{background:#19c37d;color:#fff;animation:pulse 1.4s infinite}}
@keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.1)}}}}
.mic-txt{{font-size:12px;color:#555;font-style:italic;max-width:140px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}}
</style>

<div class="pill" id="pill">
  <div class="row1">
    <textarea id="ta" rows="1" placeholder="{tx["ph"]}" onkeydown="kd(event)" oninput="grow(this)"></textarea>
    <button class="send" id="sendbtn" onclick="send()">↑</button>
  </div>
  <div class="row2">
    <button class="abtn" onclick="attach()" title="Attach photo">📎</button>
    <button class="abtn {'on' if st.session_state.call else ''}" id="callbtn" onclick="callAct()" {'disabled' if st.session_state.call else ''}>📞</button>
    <button class="abtn" onclick="stopVoice()">⏹</button>
    <span class="mic-txt" id="mictxt"></span>
    <button class="mic-btn" id="micbtn" onclick="toggleMic()" title="Voice input">🎤</button>
  </div>
</div>

<script>
var rec=null,micOn=false;

function grow(el){{
  el.style.height='auto';
  el.style.height=Math.min(el.scrollHeight,160)+'px';
  document.getElementById('sendbtn').className=el.value.trim()?'send active':'send';
}}

function kd(e){{
  if(e.key==='Enter'&&!e.shiftKey){{e.preventDefault();send()}}
}}

function send(){{
  var txt=document.getElementById('ta').value.trim();
  if(!txt) return;
  window.parent.postMessage({{type:'pd_send',text:txt}},'*');
  document.getElementById('ta').value='';
  document.getElementById('ta').style.height='auto';
  document.getElementById('sendbtn').className='send';
}}

function attach(){{
  window.parent.postMessage({{type:'pd_action',action:'attach'}},'*');
}}

function callAct(){{
  window.parent.postMessage({{type:'pd_action',action:'call'}},'*');
}}

function stopVoice(){{
  window.parent.postMessage({{type:'pd_action',action:'stop'}},'*');
}}

function toggleMic(){{micOn?stopMic():startMic()}}

function startMic(){{
  var SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){{document.getElementById('mictxt').textContent='{tx["mch"]}';return}}
  rec=new SR(); rec.lang='{lc}'; rec.continuous=false;
  rec.interimResults=true; rec.maxAlternatives=1;
  micOn=true;
  document.getElementById('micbtn').className='mic-btn on';
  document.getElementById('micbtn').textContent='⏹';
  document.getElementById('mictxt').textContent='{tx["mls"]}';
  rec.onresult=function(e){{
    var final='',interim='';
    for(var i=e.resultIndex;i<e.results.length;i++){{
      if(e.results[i].isFinal) final+=e.results[i][0].transcript;
      else interim+=e.results[i][0].transcript;
    }}
    document.getElementById('mictxt').textContent=final||interim;
    if(final){{
      document.getElementById('ta').value=final;
      grow(document.getElementById('ta'));
      setTimeout(function(){{send()}},100);
      stopMic();
    }}
  }};
  rec.onerror=function(e){{document.getElementById('mictxt').textContent='Error: '+e.error;stopMic()}};
  rec.onend=function(){{if(micOn)stopMic()}};
  try{{rec.start()}}catch(ex){{stopMic()}}
}}

function stopMic(){{
  micOn=false; if(rec)try{{rec.abort()}}catch(e){{}}
  document.getElementById('micbtn').className='mic-btn';
  document.getElementById('micbtn').textContent='🎤';
  document.getElementById('mictxt').textContent='';
}}

window.addEventListener('message',function(e){{
  if(e.data&&e.data.type==='pd_insert'){{
    document.getElementById('ta').value=e.data.text;
    grow(document.getElementById('ta'));
  }}
}});
</script>
"""

# Receive messages from HTML component via query params trick
# We use a hidden text input + form as message bus
components.html(pill_html, height=130)

st.markdown(f'<div class="ifoot">Plant Doctor AI · Offline · {tx["priv"]}</div>',unsafe_allow_html=True)
st.markdown('</div></div>',unsafe_allow_html=True)

# ── MESSAGE BUS — truly hidden via CSS ────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stForm"]:has(button[kind="formSubmit"]) {
  position:fixed!important;bottom:-9999px!important;left:-9999px!important;
  width:1px!important;height:1px!important;overflow:hidden!important;
  opacity:0!important;pointer-events:none!important;z-index:-1!important
}
</style>""", unsafe_allow_html=True)
with st.form("msg_bus",clear_on_submit=True):
    bus_text=st.text_input("bus","",key="bus_input")
    bus_action=st.text_input("act","",key="bus_action")
    bus_submit=st.form_submit_button("go")

# JS bridge: intercept postMessage and route to hidden form
components.html("""
<script>
window.addEventListener('message',function(e){
  if(!e.data||!e.data.type) return;
  var pd=window.parent.document;

  function setVal(el,val){
    var s=Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype,'value').set;
    s.call(el,val);
    el.dispatchEvent(new Event('input',{bubbles:true}));
  }

  function clickSubmit(){
    setTimeout(function(){
      var btns=pd.querySelectorAll('div[data-testid="stFormSubmitButton"] button');
      if(btns.length) btns[btns.length-1].click();
    },100);
  }

  // Find inputs by placeholder label text in the form
  function findInputs(){
    var all=pd.querySelectorAll('input[type="text"]');
    var bus=null,act=null;
    all.forEach(function(el){
      var label=el.closest('[data-testid="stTextInput"]');
      if(!label) return;
      var lbl=label.querySelector('label');
      if(lbl&&lbl.textContent.trim()==='bus') bus=el;
      if(lbl&&lbl.textContent.trim()==='act') act=el;
    });
    // fallback: last two inputs
    if(!bus&&all.length>=2) bus=all[all.length-2];
    if(!act&&all.length>=1) act=all[all.length-1];
    return {bus:bus,act:act};
  }

  if(e.data.type==='pd_send'){
    var f=findInputs();
    if(f.bus){ setVal(f.bus,e.data.text); }
    if(f.act){ setVal(f.act,''); }
    clickSubmit();
  }

  if(e.data.type==='pd_action'){
    var f=findInputs();
    if(f.bus){ setVal(f.bus,''); }
    if(f.act){ setVal(f.act,e.data.action); }
    clickSubmit();
  }
});
</script>""",height=0)

# ── PROCESS INPUTS ─────────────────────────────────────────────────────────────
if bus_submit:
    txt=bus_text.strip(); act=bus_action.strip()
    if txt:
        add("u",txt); r=get_chatbot_response(txt,st.session_state.last_dis,sel)
        add("b",r); speak_text(r[:400],sel); st.rerun()
    elif act=="call" and not st.session_state.call:
        st.session_state.call=True; g=tx["cg"]
        add("u","📞",t="cs"); add("b",g,t="call")
        speak_text(g,sel); st.rerun()
    elif act=="stop":
        stop_speaking()

if uploaded is not None:
    fid=f"{uploaded.name}_{uploaded.size}"
    if st.session_state.proc!=fid:
        st.session_state.proc=fid; image=Image.open(uploaded)
        add("u","📷",t="img",img=image)
        with st.spinner(tx["anal"]):
            qc=check_image_quality(image)
            if not qc["ok"]: add("b"," | ".join(qc["issues"]),t="err")
            else:
                lf=is_leaf(image)
                if not lf["is_leaf"]: add("b",lf["reason"],t="err")
                else:
                    dis,conf=predict(image)
                    try:
                        from modules.cloud_vision import verify_disease
                        dis = verify_disease(image, dis)
                    except Exception:
                        pass
                    sev=estimate_severity(image)
                    sev["advice"]=get_severity_advice(sev["severity"],sel)
                    hl=highlight(image); inf=get_disease_info(dis,sel)
                    age=detect_leaf_age(image,sel); st.session_state.last_dis=dis
                    
                    try:
                        db.save_diagnosis(
                            disease=dis, confidence=conf, severity=sev.get("severity", "LOW"),
                            infected_pct=sev.get("infected_percent", 0.0), 
                            plant_name=st.session_state.selected_plant,
                            language=sel
                        )
                    except Exception as e:
                        print(f"Failed to log diagnosis: {e}")
                        
                    add("b",dis,t="dx",dis=dis,conf=conf,sev=sev,
                        info=inf or {},hl=hl,orig=image,age=age)
                    add("b",tx["fup"])
                    te=(inf.get("treatment_chemical","") if inf else "")
                    speak_text(build_voice_text(dis,sev["severity"],sev["advice"],te,sel),sel)
        st.rerun()