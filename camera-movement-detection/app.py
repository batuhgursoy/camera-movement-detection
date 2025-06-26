import streamlit as st
import numpy as np
import cv2
import tempfile
import movement_detector

st.set_page_config(
    page_title="Advanced Camera Movement Detection",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 Advanced Camera Movement Detection")
st.write(
    "Upload a video file to detect and classify significant camera movements like panning, tilting, rotation, and zooming."
)

# Sidebar for advanced controls
with st.sidebar:
    st.header("⚙️ Analysis Controls")
    
    st.subheader("Sensitivity Thresholds")
    translation_threshold = st.slider(
        "Translation Threshold (pixels)",
        min_value=0.1,
        max_value=50.0,
        value=0.6,
        step=0.1,
        help="Detects camera panning or tilting. Lower values are more sensitive.",
    )
    rotation_threshold = st.slider(
        "Rotation Threshold (degrees)",
        min_value=0.1,
        max_value=10.0,
        value=0.3,
        step=0.1,
        help="Detects camera rotation. Lower values are more sensitive.",
    )
    scale_threshold = st.slider(
        "Scale/Zoom Threshold (%)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Detects camera zooming. Lower values are more sensitive to minor zooms.",
    )

    st.subheader("Noise & Stability Control")
    smoothing_window = st.slider(
        "Temporal Smoothing Window (frames)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Averages motion over N frames to reduce noise from single-frame glitches. Higher values are more stable but less responsive.",
    )

uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if frames:
        st.info(f"📹 Video loaded successfully with {len(frames)} frames. Analyzing...")

        # --- DIAGNOSTICS SECTION ---
        with st.expander("Show Raw Frame-by-Frame Analysis (for debugging)"):
            st.warning(
                "This table shows raw motion metrics calculated between each frame *before* applying smoothing and thresholds. "
                "If all values are zero or homography is not found, it indicates an issue with feature matching in the video."
            )
            
            prev_gray_debug = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            debug_data = []
            
            # Limit to first 100 frames for performance and clarity
            for i in range(1, min(len(frames), 101)):
                current_gray_debug = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                M_debug = movement_detector.find_homography_matrix(prev_gray_debug, current_gray_debug)
                
                metrics = {"frame": i, "translation": 0.0, "rotation": 0.0, "scale": 1.0, "homography_found": False}
                if M_debug is not None:
                    raw_metrics = movement_detector.analyze_homography(M_debug)
                    metrics.update(raw_metrics)
                    metrics["homography_found"] = True

                debug_data.append(metrics)
                prev_gray_debug = current_gray_debug
                
            st.dataframe(debug_data)
        # --- END DIAGNOSTICS ---

        with st.spinner("Applying advanced motion analysis... This may take a moment."):
            detected_movements = movement_detector.detect_significant_movement(
                frames,
                translation_threshold=translation_threshold,
                rotation_threshold=rotation_threshold,
                scale_threshold=scale_threshold / 100.0,  # Convert percentage to ratio
                smoothing_window=smoothing_window,
            )

        st.success(f"Analysis complete. Found {len(detected_movements)} instances of significant movement.")

        if detected_movements:
            st.write("### Detected Movement Report")
            
            # Display images in a responsive grid
            for movement in detected_movements:
                idx = movement["frame_index"]
                
                with st.container():
                    st.image(
                        frames[idx],
                        caption=f"Frame {idx}: {movement['type']}",
                        use_container_width=True,
                    )
                    st.metric(label="Movement Type", value=movement['type'])
                    metrics_cols = st.columns(3)
                    metrics_cols[0].metric(label="Translation (px)", value=f"{movement['translation']:.2f}")
                    metrics_cols[1].metric(label="Rotation (°)", value=f"{movement['rotation']:.2f}")
                    metrics_cols[2].metric(label="Scale", value=f"{movement['scale']:.3f}")
                    st.markdown("---")
        else:
            st.write("No significant camera movement was detected based on the current settings.")

    tfile.close()
