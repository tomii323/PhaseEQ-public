"""Render ordered help sections without changing application settings."""
import streamlit as st


def render_help_sections(sections):
    sections = tuple(sections)
    if not sections:
        return
    for tab, section in zip(st.tabs([section.title for section in sections]), sections):
        with tab:
            st.markdown(section.body)
