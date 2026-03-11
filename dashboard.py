"""
Admin Dashboard — Helmet Violation Detection System
Run from the project root:
    streamlit run dashboard.py
"""

import os
from groq import Groq
from datetime import datetime
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
_groq_key = os.getenv("GROQ_API_KEY", "")

def ask_gemini(question, certain_df, uncertain_df, rejected_df):
    parts = []
    if not certain_df.empty:
        parts.append(f"Certain violations:\n{certain_df.to_string(index=False)}")
    if not uncertain_df.empty:
        parts.append(f"Uncertain violations:\n{uncertain_df.to_string(index=False)}")
    if not rejected_df.empty:
        parts.append(f"Rejected (false positives):\n{rejected_df.to_string(index=False)}")
    context = "\n\n".join(parts) if parts else "No data available yet."

    prompt = (
        "You are an assistant for a traffic helmet violation detection system. "
        "Answer using only the data below. Be concise.\n\n"
        f"Data:\n{context}\n\nQuestion: {question}"
    )
    client = Groq(api_key=_groq_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CERTAIN_CSV   = os.path.join(BASE_DIR, "violation_certain.csv")
UNCERTAIN_CSV = os.path.join(BASE_DIR, "violation_uncertain.csv")
REJECTED_CSV  = os.path.join(BASE_DIR, "violation_rejected.csv")
OWNER_CSV     = os.path.join(BASE_DIR, "violations.csv")   # LP → Name/Gmail lookup

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Helmet Violation Admin",
    page_icon="🚨",
    layout="wide",
)
st.title("🚨 Helmet Violation Detection — Admin Dashboard")
st.markdown("---")

# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, header=0)
    df.columns = df.columns.str.strip()
    return df

def save_csv(df, path):
    df.to_csv(path, index=False)

def ensure_col(df, col, default=""):
    if col not in df.columns:
        df[col] = default
    return df

def lookup_owner(plate):
    """Look up owner Name and Gmail from violations.csv (columns: STT,Name,Gmail,Address,LP)."""
    owner_df = load_csv(OWNER_CSV)
    if owner_df.empty or "LP" not in owner_df.columns:
        return "Unknown", "N/A"
    match = owner_df[owner_df["LP"] == plate]
    if match.empty:
        return "Unknown", "N/A"
    row = match.iloc[0]
    return str(row.get("Name", "Unknown")), str(row.get("Gmail", "N/A"))

def show_images(rider_path, plate_path):
    c1, c2 = st.columns(2)
    with c1:
        rp = str(rider_path).strip()
        if rp and os.path.exists(rp):
            st.image(rp, caption="Rider / Scene", use_container_width=True)
        else:
            st.warning("Rider image not found.")
    with c2:
        pp = str(plate_path).strip() if pd.notna(plate_path) else ""
        if pp and os.path.exists(pp):
            st.image(pp, caption="Plate", use_container_width=True)
        else:
            st.info("No plate image available.")

# ── Load CSVs ─────────────────────────────────────────────────────────────────
certain_df   = load_csv(CERTAIN_CSV)
certain_df   = ensure_col(certain_df, "ChallanTimestamp")
uncertain_df = load_csv(UNCERTAIN_CSV)
rejected_df  = load_csv(REJECTED_CSV)

# ── Top metrics ───────────────────────────────────────────────────────────────
issued_count = 0
if not certain_df.empty and "Status" in certain_df.columns:
    issued_count = int((certain_df["Status"] == "Challan Issued").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("✅ Certain Violations", len(certain_df))
m2.metric("🎫 Challan Issued",     issued_count)
m3.metric("⚠️ Uncertain",          len(uncertain_df))
m4.metric("❌ Rejected",            len(rejected_df))
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "✅ Certain Violations",
    "⚠️ Uncertain — Needs Review",
    "❌ Rejected (Audit Trail)",
    "🤖 Ask AI",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Certain Violations
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if certain_df.empty:
        st.info("No certain violations logged yet.")
    else:
        summary_cols = [c for c in
                        ["Timestamp","Plate","Owner","Gmail","Status","ChallanTimestamp"]
                        if c in certain_df.columns]

        st.caption("Click a row to review it.")
        event = st.dataframe(
            certain_df[summary_cols],
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            key="certain_table",
        )

        selected = event.selection.rows
        if not selected:
            st.info("Select a row above to review it.")
        else:
            idx = selected[0]
            i   = certain_df.index[idx]
            row = certain_df.loc[i]

            plate      = str(row.get("Plate", "Unknown"))
            ts         = str(row.get("Timestamp", f"Record {i}"))
            status     = str(row.get("Status", ""))
            challan_ts = str(row.get("ChallanTimestamp", "")).strip()

            st.markdown("---")
            st.markdown(f"### Reviewing: **{plate}** — {ts}")

            img_col, action_col = st.columns([2, 1])

            with img_col:
                show_images(row.get("RiderImage", ""), row.get("PlateImage", ""))

            with action_col:
                for field in ["Plate","Owner","Gmail","Status"]:
                    if field in row:
                        st.write(f"**{field}:** {row[field]}")
                if challan_ts:
                    st.write(f"**Challan issued at:** {challan_ts}")

                st.markdown("---")

                # ── Correct Plate ─────────────────────────────────────
                st.markdown("**✏️ Correct Plate**")
                new_plate = st.text_input(
                    "New plate number:", value=plate,
                    key=f"correct_{i}"
                ).upper().strip()
                if st.button("💾 Save correction", key=f"save_correct_{i}"):
                    if new_plate and new_plate != plate:
                        certain_df.at[i, "Plate"] = new_plate
                        name, gmail = lookup_owner(new_plate)
                        certain_df.at[i, "Owner"] = name
                        certain_df.at[i, "Gmail"] = gmail
                        save_csv(certain_df, CERTAIN_CSV)
                        st.success(f"Plate updated to {new_plate}")
                        st.rerun()
                    else:
                        st.warning("Enter a different plate number to correct.")

                st.markdown("---")

                # ── Issue Challan ─────────────────────────────────────
                if status != "Challan Issued":
                    if st.button("🎫 Issue Challan", key=f"challan_{i}", type="primary"):
                        certain_df.at[i, "Status"]           = "Challan Issued"
                        certain_df.at[i, "ChallanTimestamp"] = \
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_csv(certain_df, CERTAIN_CSV)
                        st.success("Challan issued!")
                        st.rerun()
                else:
                    st.success(f"✅ Challan already issued at {challan_ts}")

                st.markdown("---")

                # ── Reject ────────────────────────────────────────────
                st.markdown("**❌ Reject as False Positive**")
                reason = st.text_input(
                    "Rejection reason:", key=f"reason_{i}",
                    placeholder="e.g. Rider was wearing helmet"
                )
                if st.button("❌ Reject", key=f"reject_{i}"):
                    if reason.strip():
                        rejected_row = row.to_dict()
                        rejected_row["RejectedAt"]      = \
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        rejected_row["RejectionReason"] = reason.strip()
                        rej_df = load_csv(REJECTED_CSV)
                        rej_df = pd.concat(
                            [rej_df, pd.DataFrame([rejected_row])], ignore_index=True)
                        save_csv(rej_df, REJECTED_CSV)
                        certain_df = certain_df.drop(index=i).reset_index(drop=True)
                        save_csv(certain_df, CERTAIN_CSV)
                        st.success("Rejected — moved to audit trail.")
                        st.rerun()
                    else:
                        st.error("Please enter a rejection reason.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Uncertain Violations
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if uncertain_df.empty:
        st.info("No uncertain violations pending review.")
    else:
        unc_summary_cols = [c for c in
                            ["Timestamp","Frame","Plate","Owner","Status"]
                            if c in uncertain_df.columns]

        st.caption("Click a row to review it.")
        unc_event = st.dataframe(
            uncertain_df[unc_summary_cols],
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            key="uncertain_table",
        )

        unc_selected = unc_event.selection.rows
        if not unc_selected:
            st.info("Select a row above to review it.")
        else:
            idx = unc_selected[0]
            i   = uncertain_df.index[idx]
            row = uncertain_df.loc[i]

            ts    = str(row.get("Timestamp", f"Record {i}"))
            frame = row.get("Frame", "?")

            st.markdown("---")
            st.markdown(f"### Reviewing: Frame **{frame}** — {ts}")

            img_col, action_col = st.columns([2, 1])

            with img_col:
                show_images(row.get("RiderImage", ""), row.get("PlateImage", ""))

            with action_col:
                for field in ["Frame","Plate","Owner","Status"]:
                    if field in row:
                        st.write(f"**{field}:** {row[field]}")

                st.markdown("---")

                # ── Accept ────────────────────────────────────────────
                st.markdown("**✅ Accept — Enter Correct Plate**")
                plate_input = st.text_input(
                    "Plate number:", key=f"unc_plate_{i}",
                    placeholder="e.g. MH12LC9488"
                ).upper().strip()
                if st.button("✅ Accept & Move to Certain",
                             key=f"accept_{i}", type="primary"):
                    if plate_input:
                        name, gmail = lookup_owner(plate_input)
                        new_row = {
                            "Timestamp"       : ts,
                            "Frame"           : frame,
                            "Plate"           : plate_input,
                            "Owner"           : name,
                            "Gmail"           : gmail,
                            "Status"          : "Manually Verified",
                            "ChallanTimestamp": "",
                            "RiderImage"      : row.get("RiderImage", ""),
                            "PlateImage"      : row.get("PlateImage", ""),
                        }
                        cert_df = load_csv(CERTAIN_CSV)
                        cert_df = ensure_col(cert_df, "ChallanTimestamp")
                        cert_df = pd.concat(
                            [cert_df, pd.DataFrame([new_row])], ignore_index=True)
                        save_csv(cert_df, CERTAIN_CSV)
                        uncertain_df = uncertain_df.drop(index=i).reset_index(drop=True)
                        save_csv(uncertain_df, UNCERTAIN_CSV)
                        st.success(f"Accepted as {plate_input} — moved to Certain.")
                        st.rerun()
                    else:
                        st.error("Please enter the correct plate number.")

                st.markdown("---")

                # ── Reject ────────────────────────────────────────────
                st.markdown("**❌ Reject as False Positive**")
                rej_reason = st.text_input(
                    "Rejection reason:", key=f"unc_reason_{i}",
                    placeholder="e.g. Rider was wearing helmet"
                )
                if st.button("❌ Reject", key=f"unc_reject_{i}"):
                    if rej_reason.strip():
                        rejected_row = row.to_dict()
                        rejected_row["RejectedAt"]      = \
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        rejected_row["RejectionReason"] = rej_reason.strip()
                        rej_df = load_csv(REJECTED_CSV)
                        rej_df = pd.concat(
                            [rej_df, pd.DataFrame([rejected_row])], ignore_index=True)
                        save_csv(rej_df, REJECTED_CSV)
                        uncertain_df = uncertain_df.drop(index=i).reset_index(drop=True)
                        save_csv(uncertain_df, UNCERTAIN_CSV)
                        st.success("Rejected — moved to audit trail.")
                        st.rerun()
                    else:
                        st.error("Please enter a rejection reason.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Rejected (Audit Trail)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if rejected_df.empty:
        st.info("No rejected violations yet.")
    else:
        rej_summary_cols = [c for c in
                            ["Timestamp","Frame","Plate","Owner","Status",
                             "RejectedAt","RejectionReason"]
                            if c in rejected_df.columns]

        st.caption("Click a row to view evidence.")
        rej_event = st.dataframe(
            rejected_df[rej_summary_cols],
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            key="rejected_table",
        )

        rej_selected = rej_event.selection.rows
        if not rej_selected:
            st.info("Select a row above to view its evidence.")
        else:
            idx = rej_selected[0]
            i   = rejected_df.index[idx]
            row = rejected_df.loc[i]

            ts     = str(row.get("Timestamp", f"Record {i}"))
            plate  = str(row.get("Plate", "?"))
            reason = str(row.get("RejectionReason", ""))

            st.markdown("---")
            st.markdown(f"### Evidence: **{plate}** — {ts}")

            img_col, info_col = st.columns([2, 1])
            with img_col:
                show_images(row.get("RiderImage", ""), row.get("PlateImage", ""))
            with info_col:
                for field in ["Plate","Owner","Status","RejectedAt","RejectionReason"]:
                    if field in row:
                        st.write(f"**{field}:** {row[field]}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Chatbot
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 Ask about your violation data")

    if not _groq_key:
        st.error("GROQ_API_KEY not set. Add it to your .env file.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask anything about the violations...")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_gemini(user_q, certain_df, uncertain_df, rejected_df)
                    except Exception as e:
                        answer = f"Error: {e}"
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})