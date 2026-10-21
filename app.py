"""ReliableTSR Enterprise: Secure Document Table Intelligence with TATR & Alibaba Qwen."""
import json
import io
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st
from PIL import Image, ImageDraw

from reliabletsr.auth.user_manager import UserManager
from reliabletsr.auth.session import SessionManager, UserSession
from reliabletsr.storage.sandbox import EphemeralStorage
from reliabletsr.document.pdf_processor import DocumentProcessor
from reliabletsr.detection.table_detector import TableDetector, DetectedTable
from reliabletsr.model import TATR
from reliabletsr.structure import build_grid
from reliabletsr.risk import assess
from reliabletsr.qwen.client import QwenClient
from reliabletsr.qwen.verifier import QwenTableAuditor
from reliabletsr.export.excel_exporter import TableExporter
from reliabletsr.errors import compare

# ---------------------------------------------------------------------------
# App Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ReliableTSR Enterprise",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Global Cached Resources & Session State
# ---------------------------------------------------------------------------
@st.cache_resource
def get_user_manager() -> UserManager:
    return UserManager()


@st.cache_resource
def get_session_manager() -> SessionManager:
    return SessionManager(max_idle_seconds=3600)


@st.cache_resource
def get_ephemeral_storage() -> EphemeralStorage:
    return EphemeralStorage(max_file_size_mb=15, max_docs_per_user=5)


@st.cache_resource
def get_tatr_model(local_model: Optional[str] = None, cache_dir: str = "data/model-cache", device: str = "cpu") -> TATR:
    return TATR(local_model or None, cache_dir or None, device)


@st.cache_resource
def get_table_detector(device: str = "cpu", cache_dir: str = "data/model-cache") -> TableDetector:
    return TableDetector(device=device, cache_dir=cache_dir)


def overlay_grid(image: Image.Image, cells: List[dict], highlighted=(), color: str = "lime") -> Image.Image:
    """Draw bounding boxes over reconstructed cells."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    highlight_set = set(highlighted)
    for cell in cells:
        box = cell["bbox"]
        is_highlighted = cell["id"] in highlight_set
        outline_color = "red" if is_highlighted else color
        line_width = 4 if is_highlighted else 2
        draw.rectangle(box, outline=outline_color, width=line_width)
        draw.text((box[0] + 3, box[1] + 3), cell["id"], fill=outline_color)
    return canvas


# Initialize session state keys
if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = None
if "authenticated_user" not in st.session_state:
    st.session_state["authenticated_user"] = None
if "active_document" not in st.session_state:
    st.session_state["active_document"] = None
if "detected_tables" not in st.session_state:
    st.session_state["detected_tables"] = []
if "selected_table_idx" not in st.session_state:
    st.session_state["selected_table_idx"] = 0
if "table_results" not in st.session_state:
    st.session_state["table_results"] = {}
if "openrouter_key" not in st.session_state:
    st.session_state["openrouter_key"] = ""

user_mgr = get_user_manager()
session_mgr = get_session_manager()
storage = get_ephemeral_storage()

# ---------------------------------------------------------------------------
# Authentication & Landing Screen (When Logged Out)
# ---------------------------------------------------------------------------
if not st.session_state["authenticated_user"]:
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="color: #1F4E79; margin-bottom: 0.2rem;">ReliableTSR Enterprise</h1>
            <p style="color: #666; font-size: 1.15rem; margin-top: 0;">
                Secure Document Table Intelligence · Cascaded Geometric Error Detection & Multimodal AI
            </p>
            <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
                <span style="background-color: #EBF3FB; color: #1F4E79; padding: 5px 12px; border-radius: 15px; font-size: 0.85rem; font-weight: 600;">
                    🔒 Zero-Retention In-Memory Storage
                </span>
                <span style="background-color: #EBF3FB; color: #1F4E79; padding: 5px 12px; border-radius: 15px; font-size: 0.85rem; font-weight: 600;">
                    ⚡ Laptop-Optimized (0 GPU Required)
                </span>
                <span style="background-color: #EBF3FB; color: #1F4E79; padding: 5px 12px; border-radius: 15px; font-size: 0.85rem; font-weight: 600;">
                    🤖 Alibaba Qwen2.5-VL Forensic Judge
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.write("")

    col_l, col_center, col_r = st.columns([1, 1.2, 1])
    with col_center:
        st.subheader("Secure Organization Sign In")
        login_user = st.text_input("Username", value="analyst", placeholder="Enter your username")
        login_pass = st.text_input("Password", type="password", value="Analyst@DocAI!", placeholder="Enter your password")

        col_login, col_demo = st.columns(2)
        with col_login:
            if st.button("Sign In", type="primary", use_container_width=True):
                user = user_mgr.authenticate(login_user, login_pass)
                if user:
                    session = session_mgr.create_session(user)
                    st.session_state["authenticated_user"] = user
                    st.session_state["current_session_id"] = session.session_id
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with col_demo:
            if st.button("Quick Demo Sign In", use_container_width=True):
                user = user_mgr.authenticate("demo", "DemoPass123!")
                if user:
                    session = session_mgr.create_session(user)
                    st.session_state["authenticated_user"] = user
                    st.session_state["current_session_id"] = session.session_id
                    st.rerun()

        st.caption("Default accounts: `admin` / `Admin@Secure2026!`, `analyst` / `Analyst@DocAI!`, `demo` / `DemoPass123!`")
        st.info("🛡️ **Enterprise Privacy Notice**: All uploaded documents are processed purely in volatile RAM buffers. No files are saved to unencrypted disks, and all buffers auto-destruct on session logout.")

    st.stop()

# ---------------------------------------------------------------------------
# Authenticated User Session Navigation Header
# ---------------------------------------------------------------------------
current_user = st.session_state["authenticated_user"]
current_session_id = st.session_state["current_session_id"]

head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.title("ReliableTSR Enterprise")
    st.caption(f"Authenticated as **{current_user.username}** (`{current_user.role}`) · Session: `{current_session_id[:8]}...` · Storage: **In-Memory Sandboxed (0 Disk)**")

with head_col2:
    st.write("")
    if st.button("🚪 Logout & Wipe RAM", type="secondary", use_container_width=True):
        storage.purge_session(current_session_id)
        session_mgr.terminate_session(current_session_id)
        st.session_state["authenticated_user"] = None
        st.session_state["current_session_id"] = None
        st.session_state["active_document"] = None
        st.session_state["detected_tables"] = []
        st.session_state["table_results"] = {}
        st.rerun()

# ---------------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    device_opt = st.selectbox("Compute Device", ["cpu", "cuda"], index=0, help="Runs fast on CPU (standard laptop).")
    
    st.divider()
    st.subheader("🤖 Alibaba Qwen2.5-VL")
    qwen_provider = st.selectbox("AI Provider", ["OpenRouter (Free Cloud)", "Local Ollama"], index=0)
    
    if qwen_provider.startswith("OpenRouter"):
        def _on_sidebar_key_change():
            st.session_state["openrouter_key"] = st.session_state["sidebar_key_box"].strip()

        st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state["openrouter_key"],
            key="sidebar_key_box",
            on_change=_on_sidebar_key_change,
            help="Free key from openrouter.ai/keys."
        )
        selected_model = st.selectbox(
            "Vision AI Model",
            [
                "nvidia/nemotron-3.5-super-vl:free",
                "google/gemma-4-31b-it:free",
                "qwen/qwen2.5-vl-72b-instruct",
                "qwen/qwen3-vl-8b-instruct",
            ],
            index=0,
            help="Choose an active 100% free model or flagship Qwen model."
        )
    else:
        openrouter_key = None
        selected_model = "qwen2.5-vl:3b"
        st.caption("Ollama Endpoint: `http://localhost:11434` (requires local Ollama running)")

    st.divider()
    st.subheader("📐 Model Detection Settings")
    table_detect_thresh = st.slider("Table Detection Threshold", min_value=0.3, max_value=0.9, value=0.6, step=0.05)
    structure_thresh = st.slider("Cell Object Threshold", min_value=0.2, max_value=0.8, value=0.5, step=0.05)
    local_checkpoint = st.text_input("Local TATR Weights (Optional)", "")

# ---------------------------------------------------------------------------
# Main Workspace Tabs
# ---------------------------------------------------------------------------
tab_doc, tab_crop, tab_eval, tab_security = st.tabs([
    "📄 Document Intelligence (PDF / Scan)",
    "🔍 Single Crop Analyzer",
    "📈 PubTables-1M Pilot Evaluation",
    "🛡️ Data Governance & Audit"
])

# ===========================================================================
# TAB 1: Document Intelligence (PDF -> Auto-Detect Tables -> Extract)
# ===========================================================================
with tab_doc:
    st.subheader("1. Upload Multi-Page Document or Scanned File")
    uploaded_file = st.file_uploader(
        "Upload a PDF or document image (Max 15MB)",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        key="doc_uploader"
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        if (not st.session_state["active_document"] or 
            st.session_state["active_document"].filename != file_name):
            with st.spinner("Rendering document in memory..."):
                try:
                    pages, mime = DocumentProcessor.process_bytes(file_bytes, file_name)
                    doc = storage.store_document(
                        session_id=current_session_id,
                        doc_id=file_name,
                        filename=file_name,
                        raw_bytes=file_bytes,
                        page_images=pages,
                        mime_type=mime
                    )
                    st.session_state["active_document"] = doc
                    st.session_state["detected_tables"] = []
                    st.session_state["table_results"] = {}
                    st.success(f"Loaded **{file_name}** ({len(pages)} pages) into secure memory.")
                except Exception as exc:
                    st.error(f"Document ingestion failed: {exc}")

    active_doc = st.session_state["active_document"]

    if active_doc:
        st.divider()
        st.subheader("2. Page Navigation & Automatic Table Detection")

        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            total_pages = len(active_doc.page_images)
            page_idx = st.number_input("Select Document Page", min_value=1, max_value=total_pages, value=1) - 1
            current_page_img = active_doc.page_images[page_idx]
            st.image(current_page_img, caption=f"Page {page_idx + 1} of {total_pages}", use_container_width=True)

            if st.button("🔍 Detect Tables on This Page", type="primary", use_container_width=True):
                with st.spinner("Running Table Detection (Microsoft TD)..."):
                    try:
                        detector = get_table_detector(device=device_opt)
                        detected = detector.detect(current_page_img, page_number=page_idx + 1, threshold=table_detect_thresh)
                        st.session_state["detected_tables"] = detected
                        st.session_state["selected_table_idx"] = 0
                        st.success(f"Identified {len(detected)} table boundary(ies) on page {page_idx + 1}.")
                    except Exception as exc:
                        st.error(f"Detection failed: {exc}")

        with col_p2:
            detected_tables: List[DetectedTable] = st.session_state["detected_tables"]
            if detected_tables:
                st.write(f"**Found {len(detected_tables)} table(s) on Page {page_idx + 1}**")
                table_options = [f"{t.table_id} (Confidence: {t.score:.1%})" for t in detected_tables]
                chosen_idx = st.selectbox("Choose Table to Analyze & Extract", range(len(detected_tables)), format_func=lambda i: table_options[i])
                st.session_state["selected_table_idx"] = chosen_idx

                selected_table = detected_tables[chosen_idx]
                st.image(selected_table.cropped_image, caption=f"Selected Table: {selected_table.table_id} (Bounding Box: {selected_table.bbox})", use_container_width=True)

                if st.button("⚡ Analyze Table Structure with TATR", type="primary"):
                    with st.spinner("Reconstructing cell grid & assessing structural risk..."):
                        tatr = get_tatr_model(local_checkpoint or None, device=device_opt)
                        detections = tatr.detect(selected_table.cropped_image, raw_threshold=0.05)
                        prediction = build_grid(detections, threshold=structure_thresh)
                        risk_info = assess(prediction, detections, [selected_table.cropped_image.width, selected_table.cropped_image.height])
                        
                        st.session_state["table_results"][selected_table.table_id] = {
                            "detections": detections,
                            "prediction": prediction,
                            "risk": risk_info,
                            "image": selected_table.cropped_image
                        }

        # Step 3 & 4: Structure Analysis & Qwen Verification
        if detected_tables and detected_tables[st.session_state["selected_table_idx"]].table_id in st.session_state["table_results"]:
            cur_tbl_id = detected_tables[st.session_state["selected_table_idx"]].table_id
            tbl_data = st.session_state["table_results"][cur_tbl_id]
            pred = tbl_data["prediction"]
            risk = tbl_data["risk"]
            tbl_img = tbl_data["image"]

            st.divider()
            st.subheader("3. ReliableTSR Structural Risk Assessment")

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Geometric Risk Score", f"{risk['score']:.2f}", delta="Low Risk" if not risk["possible_error"] else "High Risk", delta_color="inverse")
            with m_col2:
                st.metric("Predicted Cells", f"{len(pred['cells'])}")
            with m_col3:
                st.metric("Grid Dimensions", f"{len(pred['rows'])} Rows × {len(pred['columns'])} Cols")

            grid_col1, grid_col2 = st.columns(2)
            with grid_col1:
                st.image(
                    overlay_grid(tbl_img, pred["cells"], risk["affected_cells"]),
                    caption="Reconstructed Grid (Red boxes indicate cells affected by risk signals)",
                    use_container_width=True
                )
            with grid_col2:
                if risk["possible_error"]:
                    st.warning("⚠️ **Structural Anomaly Detected** (Possible silent failure in TATR geometry)")
                    for r in risk["reasons"]:
                        if r["severity"] >= 0.05:
                            st.markdown(f"- **{r['signal'].replace('_', ' ').title()}** (Severity {r['severity']:.2f}): {r['reason']} `Cells: {', '.join(r['cells']) or 'global'}`")
                else:
                    st.success("✅ **Clean Grid Structure**: No severe geometric conflicts or overlapping cells detected.")

            st.divider()
            st.subheader("4. Alibaba Qwen2.5-VL Multimodal Audit & Full Table Extraction")

            active_key = st.session_state.get("openrouter_key", "").strip()

            if not active_key and not qwen_provider.startswith("Local"):
                st.info(
                    "🔑 **Alibaba Qwen / Multimodal AI requires an OpenRouter API key** (100% Free, zero credit card).\n\n"
                    "1. Get your free key from [openrouter.ai/keys](https://openrouter.ai/keys)\n"
                    "2. Paste it below and press Enter:"
                )
                def _on_inline_key_change():
                    st.session_state["openrouter_key"] = st.session_state["inline_key_box"].strip()

                st.text_input(
                    "OpenRouter API Key (starts with sk-or-v1-...)",
                    type="password",
                    value=st.session_state["openrouter_key"],
                    key="inline_key_box",
                    on_change=_on_inline_key_change
                )
            elif active_key:
                if not active_key.startswith("sk-or-"):
                    st.warning("⚠️ **Key Format Warning**: OpenRouter API keys start with `sk-or-v1-...`. Please verify you copied the key from [openrouter.ai/keys](https://openrouter.ai/keys).")
                else:
                    st.success(f"🔑 API Key Connected: `{active_key[:12]}...{active_key[-4:]}`")

            qwen_col1, qwen_col2 = st.columns(2)
            with qwen_col1:
                st.write("**Option A: Forensic Error Audit**")
                st.caption("Asks Vision AI to inspect the flagged cells, identify false alarms, and explain true layout.")
                
                col_btn_qwen, col_btn_local = st.columns([1.2, 1])
                with col_btn_qwen:
                    run_qwen_audit = st.button("🤖 Run Qwen Audit", type="secondary", use_container_width=True, disabled=not active_key and not qwen_provider.startswith("Local"))
                with col_btn_local:
                    run_local_audit = st.button("🔍 Local Heuristic Audit", use_container_width=True)

                if run_qwen_audit:
                    with st.spinner(f"Consulting Multimodal AI ({selected_model})..."):
                        try:
                            client = QwenClient(
                                api_key=active_key,
                                provider="ollama" if qwen_provider.startswith("Local") else "openrouter",
                                model=selected_model
                            )
                            auditor = QwenTableAuditor(client)
                            audit_result = auditor.audit_table(tbl_img, risk, table_id=cur_tbl_id)
                            st.session_state[f"audit_{cur_tbl_id}"] = audit_result.qwen_verdict
                        except Exception as exc:
                            st.error(f"Audit failed: {exc}")

                if run_local_audit:
                    # Provide local explainability report from ReliableTSR geometry without external API
                    reasons_list = [f"• **{r['signal'].replace('_', ' ').title()}** (Severity: {r['severity']:.2f}): {r['reason']} (Cells: `{', '.join(r['cells']) or 'table-level'}`)" for r in risk.get("reasons", []) if r.get("severity", 0) >= 0.05]
                    local_report = (
                        f"### Local Geometric Audit Report for `{cur_tbl_id}`\n"
                        f"- **Table Status**: {'⚠️ High Risk of Structural Error' if risk['possible_error'] else '✅ Structurally Sound'}\n"
                        f"- **Anomaly Score**: `{risk['score']:.2f}` (Warning Threshold: `{risk['threshold']:.2f}`)\n"
                        f"- **Affected Cells**: `{', '.join(risk['affected_cells']) or 'None'}`\n\n"
                        f"**Flagged Anomaly Signals:**\n" + ("\n".join(reasons_list) if reasons_list else "None (all invariants passed).") + "\n\n"
                        f"*Tip: To verify whether these signals are true errors or stylistic false alarms (e.g. natural whitespace), enter an OpenRouter key above to run Alibaba Qwen2.5-VL.*"
                    )
                    st.session_state[f"audit_{cur_tbl_id}"] = local_report

                if f"audit_{cur_tbl_id}" in st.session_state:
                    st.info(f"**Audit Report:**\n\n{st.session_state[f'audit_{cur_tbl_id}']}")

            with qwen_col2:
                st.write("**Option B: Full Table Transcription to Excel / Markdown**")
                st.caption("Transcribes all text, numbers, and headers into a structured table ready for export.")

                if st.button("📊 Extract Complete Table (OCR + Structure)", type="primary", use_container_width=True, disabled=not active_key and not qwen_provider.startswith("Local")):
                    with st.spinner(f"Extracting table content ({selected_model})..."):
                        try:
                            client = QwenClient(
                                api_key=active_key,
                                provider="ollama" if qwen_provider.startswith("Local") else "openrouter",
                                model=selected_model
                            )
                            auditor = QwenTableAuditor(client)
                            markdown_text = auditor.extract_table(tbl_img)
                            st.session_state[f"markdown_{cur_tbl_id}"] = markdown_text
                        except Exception as exc:
                            st.error(f"Extraction failed: {exc}")

                if f"markdown_{cur_tbl_id}" in st.session_state:
                    extracted_md = st.session_state[f"markdown_{cur_tbl_id}"]
                    if extracted_md and extracted_md.strip():
                        st.markdown("**Preview Extracted Data:**")
                        
                        # Render structured table view
                        parsed_rows = TableExporter.parse_table(extracted_md)
                        if len(parsed_rows) >= 2:
                            try:
                                import pandas as pd
                                max_cols = max(len(r) for r in parsed_rows)
                                padded = [r + [""] * (max_cols - len(r)) for r in parsed_rows]
                                df = pd.DataFrame(padded[1:], columns=padded[0])
                                st.dataframe(df, use_container_width=True)
                            except Exception:
                                st.markdown(extracted_md)
                        else:
                            st.markdown(extracted_md)

                        # Export options
                        excel_bytes, mime_type = TableExporter.to_excel_bytes(extracted_md, sheet_name=cur_tbl_id)
                        csv_bytes = TableExporter.to_csv_bytes(extracted_md)

                        d_col1, d_col2 = st.columns(2)
                        with d_col1:
                            st.download_button(
                                "📥 Download Excel (.xlsx)",
                                data=excel_bytes,
                                file_name=f"{cur_tbl_id}_extracted.xlsx",
                                mime=mime_type,
                                use_container_width=True
                            )
                        with d_col2:
                            st.download_button(
                                "📄 Download CSV",
                                data=csv_bytes,
                                file_name=f"{cur_tbl_id}_extracted.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    else:
                        st.warning("⚠️ The selected model returned an empty transcription for this crop. Try selecting **`google/gemma-4-31b-it:free`** or **`qwen/qwen2.5-vl-72b-instruct`** from the sidebar and click Extract again.")

# ===========================================================================
# TAB 2: Single Crop Analyzer (Phase 1 Compatibility)
# ===========================================================================
with tab_crop:
    st.subheader("Quick Single-Crop Table Upload")
    crop_file = st.file_uploader("Upload an already-cropped table image", type=["png", "jpg", "jpeg"], key="crop_uploader")
    if crop_file and st.button("Run TATR Structure Analysis", key="btn_crop"):
        try:
            with st.spinner("Analyzing cropped table..."):
                crop_img = Image.open(io.BytesIO(crop_file.getvalue())).convert("RGB")
                tatr = get_tatr_model(local_checkpoint or None, device=device_opt)
                detections = tatr.detect(crop_img, raw_threshold=0.05)
                pred = build_grid(detections, threshold=structure_thresh)
                risk = assess(pred, detections, [crop_img.width, crop_img.height])

                c1, c2 = st.columns(2)
                c1.image(overlay_grid(crop_img, pred["cells"], risk["affected_cells"]), caption="Predicted Grid", use_container_width=True)
                with c2:
                    st.metric("Risk Score", f"{risk['score']:.2f}")
                    st.write("Possible structural error" if risk["possible_error"] else "No warning at current threshold")
                    for r in risk["reasons"]:
                        if r["severity"] >= 0.05:
                            st.warning(f"**{r['signal'].replace('_', ' ').title()}**: {r['reason']}")
        except Exception as exc:
            st.error(f"Failed to analyze cropped image: {exc}")

# ===========================================================================
# TAB 3: PubTables-1M Evaluation (Existing Benchmark Viewer)
# ===========================================================================
with tab_eval:
    st.subheader("PubTables-1M Benchmark Evaluation & Pilot Report")
    
    # 1. Display Official Measured Pilot Report if available
    report_path = Path("data/evaluation.json")
    if not report_path.exists():
        report_path = Path("examples/evaluation_report.json")
    
    if report_path.exists():
        st.markdown("### 📊 Measured Pilot Benchmark Metrics (400 PubTables Test Set)")
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        final_risk = report_data.get("final_risk", {})
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("F1-Score", f"{final_risk.get('f1', 0):.1%}", help="Harmonic mean of precision and recall")
        m2.metric("Error Recall", f"{final_risk.get('recall', 0):.1%}", help="% of true structural mistakes caught")
        m3.metric("Precision", f"{final_risk.get('precision', 0):.1%}", help="% of flagged warnings that were actual errors")
        m4.metric("False Alarm Rate", f"{final_risk.get('false_alarm_rate', 0):.1%}")

        st.write("")
        st.write("**Recall by Error Category (How well mistakes are detected):**")
        by_type = report_data.get("by_error_type", {})
        if by_type:
            type_rows = [
                {
                    "Error Type": k.replace("_", " ").title(),
                    "Test Tables": v.get("tables", 0),
                    "ReliableTSR Recall": f"{v.get('risk_recall', 0):.1%}",
                    "TATR Baseline Recall": f"{v.get('confidence_recall', 0):.1%}"
                }
                for k, v in by_type.items()
            ]
            import pandas as pd
            st.dataframe(pd.DataFrame(type_rows), use_container_width=True)

    st.divider()
    st.markdown("### 🔍 Individual Table Benchmark Records")

    eval_files = sorted(Path("data/results").rglob("*.json"))
    sample_example = Path("examples/sample_real_result.json")
    
    if not eval_files and sample_example.exists():
        eval_files = [sample_example]

    if not eval_files:
        st.info(
            "No saved per-table evaluation records found in `data/results`.\n\n"
            "To generate your own full pilot dataset from PubTables-1M:\n"
            "```powershell\n"
            "python -m reliabletsr.cli prepare --data data --count 400 --dev-count 250 --seed 1701\n"
            "python -m reliabletsr.cli infer --data data --out data/results --device cpu\n"
            "python -m reliabletsr.cli evaluate --results data/results --out data/evaluation.json\n"
            "```"
        )
    else:
        chosen_eval = st.selectbox(
            "Select Benchmark Record to Inspect",
            eval_files,
            format_func=lambda p: str(p.relative_to(Path("data/results")) if "data" in str(p) and Path("data/results").exists() and p.is_relative_to(Path("data/results")) else p.name)
        )
        record = json.loads(chosen_eval.read_text(encoding="utf-8"))
        st.write(f"Record: **{chosen_eval.name}** · Group: **{record.get('group', 'evaluation')}** · Confirmed Ground-Truth Errors: **{len(record.get('comparison', {}).get('errors', []))}**")
        
        try:
            with Image.open(record["image"]) as eval_img:
                ec1, ec2 = st.columns(2)
                ec1.image(overlay_grid(eval_img, record["prediction"]["cells"], record["risk"]["affected_cells"]), caption="TATR Prediction & Anomaly Warnings", use_container_width=True)
                ec2.image(overlay_grid(eval_img, record["ground_truth"]["grid"]["cells"], color="deepskyblue"), caption="PubTables-1M Ground Truth Grid", use_container_width=True)
        except Exception:
            st.caption("*(Original underlying image file is outside the current repository path)*")

        if record.get("comparison", {}).get("errors"):
            st.write("**Confirmed Structural Discrepancies (Hungarian Matching vs GT):**")
            err_rows = [{"Error Type": e["type"], "GT cells": ", ".join(e["gt_cells"]), "Predicted cells": ", ".join(e["pred_cells"]), "Ambiguous": e.get("ambiguous", False), "Evidence": json.dumps(e["evidence"])} for e in record["comparison"]["errors"]]
            import pandas as pd
            st.dataframe(pd.DataFrame(err_rows), use_container_width=True)

# ===========================================================================
# TAB 4: Data Governance & Security
# ===========================================================================
with tab_security:
    st.subheader("Enterprise Data Governance & Zero-Retention Audit")
    
    sec_col1, sec_col2 = st.columns(2)
    with sec_col1:
        st.markdown("""
        ### Active Security Controls
        - **In-Memory Sandboxing**: All multi-page documents, pages, and crops reside in volatile memory (`io.BytesIO`).
        - **No Unencrypted Disk Footprint**: Zero document payload is persisted to temporary or permanent disk caches.
        - **Session Multi-Tenancy**: Data structures are strictly isolated by session token.
        - **Cryptographic Hashing**: User passwords authenticated via PBKDF2-HMAC-SHA256 (100,000 iterations).
        """)

    with sec_col2:
        st.markdown("### Active Session Resource Monitor")
        active_docs = storage.list_documents(current_session_id)
        st.write(f"- Active Documents in RAM: **{len(active_docs)}**")
        for d in active_docs:
            st.write(f"  • `{d.filename}` ({d.size_bytes / 1024:.1f} KB, {len(d.page_images)} pages)")

        st.write("")
        if st.button("🗑️ Emergency Session Wipe (Purge All Data)", type="primary"):
            storage.purge_session(current_session_id)
            st.session_state["active_document"] = None
            st.session_state["detected_tables"] = []
            st.session_state["table_results"] = {}
            st.success("Session memory has been securely shredded.")
            st.rerun()
