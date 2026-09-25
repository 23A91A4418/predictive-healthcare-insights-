import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def render_metric_card(title, value, delta=None, help_text=None, color="#2E86C1"):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(240,244,248,0.9));
            border-left: 5px solid {color};
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 15px;
        ">
            <span style="font-size: 0.9rem; color: #555; font-weight: 600;">{title}</span>
            <div style="font-size: 1.8rem; font-weight: 700; color: #111; margin-top: 5px;">{value}</div>
            {f'<div style="font-size: 0.85rem; color: {"#27AE60" if "Save" in str(delta) or "High" in str(delta) else "#E74C3C"}; margin-top: 4px;">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_header():
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 25px 30px;
            border-radius: 12px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">Predictive Healthcare & Cost-Effectiveness System</h1>
            <p style="margin: 8px 0 0 0; font-size: 1.05rem; opacity: 0.9;">
                Clinical Decision Support, SHAP Explainable AI, and Hospital Financial Impact Analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
